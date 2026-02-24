# H2D / D2H PCIe Socket — Technical Report

---

## Table of Contents

1. [Background — What Is a Socket?](#1-background--what-is-a-socket)
   - 1.1 System Topology and PCIe Link Asymmetry
   - 1.2 The TT-Metal Software Stack — Where Sockets Fit
   - 1.3 Blackhole Chip at a Glance
   - 1.4 Training & Inference Use Cases for PCIe Sockets
2. [H2D Socket — Host to Device](#2-h2d-socket--host-to-device)
   - 2.1 HOST\_PUSH Mode
   - 2.2 DEVICE\_PULL Mode
3. [D2H Socket — Device to Host](#3-d2h-socket--device-to-host)
4. [Flow Control Protocol (shared)](#4-flow-control-protocol-shared)
5. [Benchmark Suite Overview](#5-benchmark-suite-overview)
6. [Results Charts](#6-results-charts)
   - 6.1 D2H Throughput
   - 6.2 D2H Latency
   - 6.3 H2D Throughput
   - 6.4 H2D Latency
   - 6.5 Ping / Jitter
   - 6.6 Multi-Chip Throughput
7. [Latency Methodology](#7-latency-methodology)
   - 7.1 What "round-trip latency" means here
   - 7.2 Measurement approach — device-side cycle counters
   - 7.3 Warmup
   - 7.4 Reporting statistics
8. [Throughput Methodology](#8-throughput-methodology)
   - 8.1 What "throughput" means here
   - 8.2 Measurement approach
9. [Sweep Parameters](#9-sweep-parameters)
10. [Benchmark Tests (one paragraph each)](#10-benchmark-tests-one-paragraph-each)
11. [Key Formulas and Constants](#11-key-formulas-and-constants)
12. [Interpreting Results](#12-interpreting-results)
13. [Running the Benchmarks](#13-running-the-benchmarks)
14. [Appendix — Detailed Parameter-Space Charts](#14-appendix--detailed-parameter-space-charts)

---

## 1. Background — What Is a Socket?

A **socket** is a streaming FIFO abstraction for moving data across the PCIe link between a host CPU and a Tenstorrent AI core. It hides the low-level details of PCIe TLB mapping, NOC address encoding, and flow-control signaling behind a simple `write` / `read` + `barrier` API.

> **Note — two different things are called "sockets" in this repo:**
> - **H2D/D2H PCIe sockets** (this document) — move data between the host CPU and a device core over PCIe. Implemented in `tt_metal/api/tt-metalium/experimental/sockets/`.
> - **TT-Fabric / Ethernet sockets** — move tensor data between AI cores across chips via Ethernet links. Documented in [`tech_reports/TT-Fabric/TT-Fabric-Architecture.md`](../../../tech_reports/TT-Fabric/TT-Fabric-Architecture.md) and [`tech_reports/Programming_Multiple_Meshes/`](../../../tech_reports/Programming_Multiple_Meshes/Programming_Multiple_Meshes.md).
>
> These are completely independent mechanisms. This document covers only the PCIe variant.

Two socket types are relevant here:

| Type | Direction | Host-side API | Device-side API |
|------|-----------|---------------|-----------------|
| `H2DSocket` | Host → Device | `write()`, `barrier()` | `SocketReceiverInterface` |
| `D2HSocket` | Device → Host | `read()`, `barrier()` | `SocketSenderInterface` |

Both types use a **circular FIFO** backed by pinned host memory (memory that the kernel guarantees will not be swapped or moved, and that has been mapped through the vIOMMU so the device can address it by its PCIe address). This is a hard system requirement: **vIOMMU must be enabled** for socket transfers to work.

The FIFO is parameterised by two quantities that directly control performance:

- **Page size** — the unit of transfer, in bytes. On Blackhole, the NOC PCIe read alignment is **64 B** and the write alignment is **16 B** (see [`tech_reports/Blackhole/BlackholeBringUpProgrammingGuide.md`](../../../tech_reports/Blackhole/BlackholeBringUpProgrammingGuide.md)). The minimum page size is therefore **64 B**, which is why the benchmark sweep starts there. Larger pages amortise per-page overhead but require bigger L1 allocations on the device.
- **FIFO (socket buffer) size** — the total capacity of the ring buffer, in bytes. A larger FIFO allows the sender to get further ahead of the receiver before back-pressure kicks in, which is critical for high throughput over high-latency PCIe.

---

### 1.1 System Topology and PCIe Link Asymmetry

Not all chips on a Tenstorrent tray are equal from a host-connectivity standpoint. In a 32-chip Blackhole Galaxy system, **all 32 chips are directly MMIO-mapped** — each has its own physical PCIe connection to the host root complex. However, chips fall into two classes based on the **bandwidth** of that connection:

| Class | Count | PCIe Generation & Width | Theoretical BW | Measured D2H Peak |
|-------|-------|------------------------|----------------|-------------------|
| High-bandwidth (ASIC 6 per tray) | 4 | **Gen 4 × 8** | ~16 GB/s | **~15.1 GB/s** |
| Low-bandwidth (all others) | 28 | **Gen 1 × 1** | ~0.25 GB/s | **~0.21 GB/s** |

The **4 high-bandwidth chips** (one per tray, ASIC Location 6) have a full Gen 4 ×8 link to the host PCIe root complex. Data written by the device kernel over NOC reaches host RAM in a single PCIe hop at full link bandwidth.

The **28 low-bandwidth chips** each have their own direct (not tunnelled) Gen 1 ×1 link to the host. There is no chip-to-chip relay — every chip's NOC write goes directly to the host via its own PCIe lane. The narrow link is a PCIe physical-layer constraint, not a routing constraint. Their peak host-facing bandwidth is roughly **70× lower** than the high-bandwidth chips regardless of page size, FIFO size, or transfer mode.

> **This asymmetry is the single most important architectural fact for training job placement.**
> Any workload that requires high-bandwidth streaming to or from the host — gradient checkpointing, activation offloading, weight streaming — must target one of the 4 high-bandwidth chips (ASIC 6 per tray). The 28 low-bandwidth chips are capped at ~0.21 GB/s for host-facing socket I/O regardless of tuning.

These numbers come from the benchmarks in this report:

```
D2H throughput, Tray 1 ASIC 6 (Gen 4 ×8):
  64 B pages → 0.30 GB/s   |   1 KB pages → 4.80 GB/s
   2 KB pages → 9.60 GB/s  |   4 KB pages → 14.92 GB/s  ← knee
  ≥16 KB pages → ~15.05 GB/s (link saturation)

D2H throughput, Tray 1 ASIC 1 (Gen 1 ×1):
  All page sizes → 0.14–0.21 GB/s  (hard ceiling, does not scale with page size)
```

This asymmetry is why the single-chip benchmarks target **Tray 1, ASIC Location 6** — the highest-bandwidth chip — and why the multi-chip benchmark sweeps all 32 chips to expose the full system-wide spread.

### Related documentation in this repo

| Document | Relevance |
|----------|-----------|
| [`tech_reports/Blackhole/BlackholeBringUpProgrammingGuide.md`](../../../tech_reports/Blackhole/BlackholeBringUpProgrammingGuide.md) | Blackhole chip specs: Tensix grid (13×10 compute), L1 (1464 KB + data cache), DRAM (~4 GB × 8 banks), NOC alignment constraints. Essential reading for understanding the hardware limits that shape the benchmark curves. |
| [`tech_reports/EthernetMultichip/BasicEthernetGuide.md`](../../../tech_reports/EthernetMultichip/BasicEthernetGuide.md) | Multi-chip topology and MMIO concepts (Wormhole-era). Note: in Wormhole only a subset of chips were MMIO-mapped; in Blackhole **all 32 chips** have direct PCIe connections. The relevant Blackhole asymmetry is PCIe link width/generation (Gen 4 ×8 vs Gen 1 ×1), not MMIO vs non-MMIO. |
| [`tech_reports/TT-Fabric/TT-Fabric-Architecture.md`](../../../tech_reports/TT-Fabric/TT-Fabric-Architecture.md) | TT-Fabric Ethernet sockets (chip-to-chip via Ethernet, **not** PCIe). Do not confuse with H2D/D2H PCIe sockets. |
| [`tech_reports/Programming_Multiple_Meshes/Programming_Multiple_Meshes.md`](../../../tech_reports/Programming_Multiple_Meshes/Programming_Multiple_Meshes.md) | Multi-mesh pipeline parallelism using Ethernet sockets. Shows how H2D/D2H PCIe sockets fit into a larger distributed training picture. |
| [`tech_reports/memory/allocator.md`](../../../tech_reports/memory/allocator.md) | L1 and DRAM allocation, alignment constraints. Relevant when choosing page sizes that respect L1 budget and NOC alignment. |

---

### 1.2 The TT-Metal Software Stack — Where Sockets Fit

Tenstorrent's software stack is layered. Understanding where H2D/D2H PCIe sockets live prevents confusion about which mechanism to use at which level of abstraction.

```
┌──────────────────────────────────────────────────────────────────────┐
│  Applications: vLLM, custom training loops, inference servers        │
├──────────────────────────────────────────────────────────────────────┤
│  TT-NN: PyTorch-like tensor op library  (ttnn.* APIs)                │
├──────────────────────────────────────────────────────────────────────┤
│  TT-Distributed / TT-Mesh: multi-host, multi-mesh abstraction        │
│  (MeshDevice, DistributedBuffer, MeshWorkload, MeshTrace)            │
├─────────────────────────────────────────────────────────────────────┤
│  TT-Metalium: single-chip programming model                         │
│  (Kernels, Programs, MeshWorkload, Command Queues)                  │
│  ┌──────────────────────────┐  ┌────────────────────────────────┐  │
│  │  H2D / D2H PCIe Sockets  │  │  TT-Fabric: inter-chip         │  │
│  │  (experimental API)      │  │  Ethernet routing firmware +   │  │
│  │  ◄── THIS DOCUMENT       │  │  Ethernet sockets              │  │
│  └──────────────────────────┘  └────────────────────────────────┘  │
├──────────────────────────────────────────────────────────────────────┤
│  Hardware: Blackhole ASIC + PCIe root complex + host RAM             │
└──────────────────────────────────────────────────────────────────────┘
```

**Key distinctions:**

| Mechanism | Layer | Transport | Typical use |
|-----------|-------|-----------|-------------|
| **H2D/D2H PCIe sockets** (this doc) | Metalium experimental | PCIe (host ↔ one chip) | Bulk data ingestion/egress; training data loaders; host-side logging |
| **TT-Fabric Ethernet sockets** | TT-Fabric firmware | Ethernet (chip ↔ chip) | Tensor shards between mesh stages; pipeline parallelism |
| **MeshCommandQueue (CQ)** | TT-Metalium runtime | PCIe (host → all chips) | Program dispatch; small weight updates; non-streaming transfers |
| **TT-Distributed sockets** | TT-Distributed | Ethernet + TCP/IP (host ↔ host) | Multi-host multi-mesh pipeline; inter-process tensor movement |

H2D/D2H PCIe sockets are a **low-level experimental API** — they bypass the CQ and the TT-NN tensor abstractions entirely. They are not yet exposed at the TT-NN or TT-Distributed layer. They are the right tool when you need **maximum streaming throughput or minimum latency between a host process and a specific device core**, and you are willing to write the device-side kernel yourself.

---

### 1.3 Blackhole Chip at a Glance

Training teams need to know the chip's compute and memory profile to reason about whether PCIe I/O is a bottleneck relative to compute and DRAM.

| Resource | Wormhole N150 | Blackhole (P150) |
|----------|---------------|------------------|
| Tensix grid | 8 × 10 (8 × 8 compute) | 14 × 10 (13 × 10 compute) |
| L1 per core | 1464 KB | 1464 KB + small write-through data cache |
| DRAM | 12 banks × 1 GB | 8 banks × ~4 GB (~32 GB total) |
| DRAM bandwidth (theoretical) | ~288 GB/s | higher total, comparable per-bank |
| Ethernet cores | 16 (1 × RISC-V, 256 KB L1) | 14 (2 × RISC-V, 512 KB L1) |
| AI clock | 1.0 GHz | **1.35 GHz** |
| Peak compute (BF8 LoFi) | ~190 TFLOPS | **~580 TFLOPS** |
| NOC PCIe read alignment | 32 B | **64 B** (sets minimum socket page size) |
| NOC PCIe write alignment | 16 B | 16 B |

Sources: [`BlackholeBringUpProgrammingGuide.md`](../../../tech_reports/Blackhole/BlackholeBringUpProgrammingGuide.md), [`GEMM_FLOPS.md`](../../../tech_reports/GEMM_FLOPS/GEMM_FLOPS.md).

> **PCIe bandwidth note (Blackhole-specific):** In a 32-chip Blackhole Galaxy, the 4 high-bandwidth chips (ASIC 6 per tray) have a **Gen 4 ×8** link (~16 GB/s theoretical, ~15.1 GB/s measured D2H). The 28 low-bandwidth chips each have a **Gen 1 ×1** link (~0.25 GB/s theoretical, ~0.21 GB/s measured). See **§1.1** for full details. WH N150 PCIe bandwidth is not directly comparable because N150 is a single-chip card with a different system topology.

---

### 1.4 Training & Inference Use Cases for PCIe Sockets

The following table maps common training and inference scenarios to the appropriate socket direction and the expected bottleneck.

| Scenario | Direction | Driver | Bottleneck |
|----------|-----------|--------|------------|
| **Training data ingestion** — loading tokenised batches from host dataloader into device L1/DRAM before each step | **H2D** | Host-side data pipeline writes to socket | PCIe Gen 4 ×8 ceiling (~16 GB/s on high-bandwidth chips). Choose `HOST_PUSH` for lowest latency, `DEVICE_PULL` to offload CPU. |
| **Activation / gradient offloading** — streaming activations or gradients out to host RAM to free device DRAM (e.g., during FSDP or gradient checkpointing) | **D2H** | Device kernel writes to host pinned buffer | PCIe Gen 4 ×8 (~16 GB/s, high-bandwidth chips only). Low-bandwidth chips are limited to ~0.21 GB/s — **do not use low-bandwidth chips for offloading**. |
| **Loss / logit collection** — reading per-step loss scalars or logit tensors from the device for host-side logging or early stopping | **D2H** | Device kernel writes small result tensors | Latency-bound (small pages). Use large page sizes even for small payloads (pad to 4 KB+) to amortise protocol overhead. |
| **Weight streaming** — pulling model weights from a host model store directly into device L1/DRAM at inference time (no pre-load) | **H2D** | Host memory manager writes to socket | PCIe bandwidth. Tune FIFO size to hide PCIe round-trip latency; use `DEVICE_PULL` to let the device DMA concurrently with compute. |
| **Pipeline stage I/O** — feeding the **first stage** of a pipeline-parallel model from a CPU-resident dataset server | **H2D** | Dataset server writes to socket into Stage 0 device | Similar to data ingestion. The socket forms the CPU→device boundary of the pipeline. Downstream stage-to-stage communication should use **Ethernet sockets** (not PCIe sockets). |
| **Telemetry / profiling streams** — continuously streaming device-side cycle counter data or custom metrics to a host monitoring process | **D2H** | Device writer kernel sends fixed-size telemetry records | Very latency-sensitive. Use small, fixed page sizes and a large FIFO. Run on a high-bandwidth chip (ASIC 6). |

> **Rule of thumb:** Any scenario that requires moving more than a few MB per second between host and device must land on one of the **4 high-bandwidth chips** (ASIC 6 per tray, Gen 4 ×8 PCIe). All 28 low-bandwidth chips are limited to ~0.21 GB/s for host-facing socket I/O, which is insufficient for streaming training data or large activation offloads at training speed.

---

## 2. H2D Socket — Host to Device

An `H2DSocket` streams data from the host CPU to a single core on the device. Two physical transfer modes are available; both share the same flow control protocol but differ in _who issues the DMA_.

### 2.1 HOST\_PUSH Mode

```
Host CPU
   │
   │  UMD TLB write  (host → device L1 FIFO)
   ▼
Device L1  ──[socket FIFO]──► Device kernel consumes page
```

The host CPU writes data directly into the device's L1 memory through a TLB-mapped PCIe write. After writing one page, the host updates the `bytes_sent` counter in host-pinned memory. The device kernel polls `bytes_sent` (via `socket_wait_for_pages`) and processes pages as they arrive.

**Acknowledgement path:** After consuming a page, the device calls `socket_notify_sender`, which issues a PCIe NOC write from the device back to the `bytes_acked` field in host-pinned memory. The host's `write()` call blocks on `bytes_acked` when the FIFO is full.

Host API:
```cpp
auto socket = H2DSocket(mesh_device, recv_core, BufferType::L1, fifo_size, H2DMode::HOST_PUSH);
socket.set_page_size(page_size);
socket.write(data_ptr, num_pages);  // blocks if FIFO full
socket.barrier();                   // waits until device has acked all data
```

Device kernel: `h2d_throughput_host_push.cpp` / `h2d_socket_data_ping_host_push.cpp`

### 2.2 DEVICE\_PULL Mode

```
Host CPU
   │
   │  writes data to pinned host memory  (no device involvement yet)
   │
   │  updates bytes_sent in pinned memory
   ▼
Device kernel polls bytes_sent, then issues
   NOC read ──► PCIe ──► pinned host memory
               (device pulls data into its L1)
```

The host writes data to a pinned host buffer (not into device L1) and then updates `bytes_sent`. The device kernel, after detecting data is available via `socket_wait_for_pages`, issues a **NOC read** from the pinned host address into its L1. This is a device-driven DMA.

This mode adds an extra hop (device must read from host RAM rather than having the host push directly into L1), but it offloads the PCIe bandwidth pressure from the host CPU and can be more efficient for large-scale streaming where the host is CPU-bottlenecked.

**Critical implementation detail — chunked NOC reads:**
NOC transactions have a maximum burst size (`NOC_MAX_BURST_SIZE`). For page sizes larger than this limit, the device kernel must loop, issuing multiple read commands per page:

```cpp
// From h2d_throughput_device_pull.cpp / h2d_socket_data_ping_device_pull.cpp
while (page_bytes_remaining) {
    uint32_t chunk_bytes = std::min(page_bytes_remaining, max_noc_burst_bytes);
    noc_read_with_state<...>(NOC_INDEX, pcie_xy_enc, page_src_addr, page_dst_addr, chunk_bytes);
    page_src_addr += chunk_bytes;
    page_dst_addr += chunk_bytes;
    page_bytes_remaining -= chunk_bytes;
}
noc_async_read_barrier();  // wait for all chunks to complete
```

Without this chunking, page sizes ≥ `NOC_MAX_BURST_SIZE` cause hangs. This was the root cause of the instability seen with 32KB and 64KB pages in early benchmark runs.

**Acknowledgement path:** Identical to HOST\_PUSH — the device issues a NOC write back to `bytes_acked` in host memory via `socket_notify_sender`.

Host API: identical to HOST\_PUSH, just pass `H2DMode::DEVICE_PULL` to the constructor.

Device kernel: `h2d_throughput_device_pull.cpp` / `h2d_socket_data_ping_device_pull.cpp`

---

## 3. D2H Socket — Device to Host

A `D2HSocket` streams data from a device core to the host. The device is always the initiator.

```
Device kernel
   │
   │  socket_reserve_pages() — spin-wait until FIFO has space
   │
   │  NOC write ──► PCIe ──► pinned host FIFO buffer
   │                          (device → host RAM)
   │
   │  socket_notify_receiver() — NOC write to update bytes_sent in host memory
   ▼
Host CPU polls bytes_sent, copies data out of pinned buffer,
then updates bytes_acked to free FIFO space
```

The device kernel is responsible for writing data into the pinned host FIFO via NOC writes. The critical address computation is:

```cpp
// From pcie_socket_sender.cpp
uint64_t pcie_data_addr =
    ((static_cast<uint64_t>(data_addr_hi) << 32) | sender_socket.downstream_fifo_addr)
    + sender_socket.write_ptr;
```

`write_ptr` is a relative offset within the FIFO, so the full 64-bit PCIe address is constructed by combining the high 32-bit address word from the socket config with the FIFO base address and the current write pointer.

Like H2D DEVICE\_PULL, large pages require chunked NOC writes to stay within burst limits:

```cpp
while (page_bytes_remaining) {
    uint32_t chunk_bytes = std::min(page_bytes_remaining, max_noc_burst_bytes);
    noc_wwrite_with_state<...>(noc_index, page_src_addr, pcie_xy_enc, page_dst_addr, chunk_bytes, 1);
    page_src_addr += chunk_bytes;
    page_dst_addr += chunk_bytes;
    page_bytes_remaining -= chunk_bytes;
}
```

Host API:
```cpp
auto socket = D2HSocket(mesh_device, sender_core, fifo_size);
socket.set_page_size(page_size);
socket.read(data_ptr, num_pages);  // blocks until data available, then acks
socket.barrier();                  // waits until device has seen all acks
```

Device kernel: `pcie_socket_sender.cpp`

---

## 4. Flow Control Protocol (shared)

Both directions use the same **bytes\_sent / bytes\_acked** credit protocol. It is asymmetric: the sender tracks `bytes_sent`, the receiver tracks `bytes_acked`. Both fields live in **host-pinned memory** accessible by both sides via PCIe.

```
Sender                          Receiver
──────────────────────────────────────────────────────
spin-wait: fifo_size - (bytes_sent - bytes_acked) ≥ page_size
write data into FIFO
bytes_sent += page_size
notify_receiver()  ─────────────────────────────►  bytes_sent updated in pinned memory
                                                    detects new data
                                                    copies / consumes data
                                                    bytes_acked += page_size
◄───────────────────────────────── notify_sender()  bytes_acked updated in pinned memory
```

- **FIFO is full** when `bytes_sent - bytes_acked == fifo_size`. The sender spins.
- **FIFO is empty** when `bytes_sent == bytes_acked`. The receiver spins.
- Both the `bytes_sent` and `bytes_acked` fields are written via NOC + PCIe, not regular load/store, so they require a cache invalidation (`invalidate_l1_cache()`) before reading on the device side.

---

## 5. Benchmark Suite Overview

All benchmarks live in `tests/tt_metal/distributed/test_hd_sockets.cpp` and run under the `HDSocketFixture` Google Test fixture.

| Test Name | Direction | What It Measures |
|-----------|-----------|-----------------|
| `D2HSocketThroughputBenchmark` | D2H | Steady-state bulk throughput |
| `D2HSocketLatencyBenchmark` | D2H | Per-iteration round-trip latency (with data DMA) |
| `D2HSocketPingBenchmark` | D2H | Pure signalling round-trip (no data DMA) |
| `D2HSocketMultiChipMaxThroughputBenchmark` | D2H | Peak throughput across all MMIO chips on system |
| `H2DSocketThroughputBenchmark` | H2D | Steady-state bulk throughput (both modes) |
| `H2DSocketMultiChipMaxThroughputBenchmark` | H2D | Peak DEVICE_PULL throughput across all chips (256 KB pages, FIFO 256 KB–1 MB) |
| `H2DSocketLatencyBenchmark` | H2D | Per-iteration round-trip latency (both modes) |
| `H2DSocketPingBenchmark` | H2D | Pure signalling round-trip (both modes) |

**Device targeting:** All single-chip benchmarks run on a standardised target: **Tray 1, ASIC Location 6**, selected using `get_target_benchmark_worker_core()`. ASIC 6 is one of the 4 high-bandwidth chips with a Gen 4 ×8 PCIe link (see **§1.1**) and gives the highest, most repeatable throughput numbers. Running on a low-bandwidth chip would cap throughput at ~0.21 GB/s regardless of configuration. The multi-chip benchmark additionally sweeps every chip on the system to give a system-wide picture.

---

## 6. Results Charts

> Place the generated `.png` files in a `charts/` subdirectory alongside this document.
> Run `analyze_d2h_throughput.py` with the appropriate flag to produce each file — see [§13 Running the Benchmarks](#13-running-the-benchmarks).

---

### 6.1 D2H Throughput

**`d2h_throughput.png`** — Throughput (GB/s) vs page size at the maximum FIFO size. Each line is one total-transfer-data size (16 KB → 1 GB). Shows how throughput saturates as pages get larger and as more data is moved.

![D2H Throughput vs Page Size](charts/d2h_throughput.png)

---

**`d2h_tp_vs_fifo.png`** — Throughput vs socket FIFO size at the maximum total-data size. Each line is one page size. The key chart for choosing a FIFO size: throughput climbs steeply with FIFO size then plateaus once back-pressure disappears.

![D2H Throughput vs FIFO Size](charts/d2h_tp_vs_fifo.png)

---

### 6.2 D2H Latency

**`d2h_latency.png`** — Round-trip latency (µs) vs page size, one line per FIFO size. p50 is shown as a solid line; min/max as dashed. Log-log scale makes both the protocol-overhead floor (small pages) and the DMA-time slope (large pages) visible.

![D2H Round-Trip Latency vs Page Size](charts/d2h_latency.png)

---

### 6.3 H2D Throughput

**`h2d_throughput.png`** — H2D throughput vs page size for both `HOST_PUSH` and `DEVICE_PULL`, at the maximum FIFO and maximum total-data size. The primary chart for comparing the two transfer modes head-to-head on throughput.

![H2D Throughput vs Page Size — HOST_PUSH vs DEVICE_PULL](charts/h2d_throughput.png)

---

**`h2d_tp_vs_fifo.png`** — H2D throughput vs FIFO size, `HOST_PUSH` (left) and `DEVICE_PULL` (right) in separate panels. Each line is one page size. Shows whether the two modes need the same FIFO depth to reach their respective plateaus.

![H2D Throughput vs FIFO Size — HOST_PUSH vs DEVICE_PULL](charts/h2d_tp_vs_fifo.png)

---

### 6.4 H2D Latency

**`h2d_latency.png`** — H2D round-trip latency vs page size, `HOST_PUSH` vs `DEVICE_PULL` overlaid on one axis (p50 solid, min/max dashed). The single most useful chart for mode selection: shows which mode has lower latency and how the gap evolves with page size.

![H2D Round-Trip Latency — HOST_PUSH vs DEVICE_PULL](charts/h2d_latency.png)

---

### 6.5 Ping / Jitter

**`d2h_ping_timeseries.png`** — D2H pure-signalling latency plotted iteration-by-iteration (no data DMA). Exposes tail-latency spikes from OS scheduler interference, PCIe power-state transitions, or NUMA effects. p50 and mean reference lines are overlaid.

![D2H Pure Ping: Per-Iteration Latency](charts/d2h_ping_timeseries.png)

---

**`h2d_ping_timeseries.png`** — H2D pure-signalling latency per iteration, `HOST_PUSH` and `DEVICE_PULL` overlaid. Directly compares the flow-control protocol overhead between the two modes with no DMA noise.

![H2D Pure Ping: Per-Iteration Latency — HOST_PUSH vs DEVICE_PULL](charts/h2d_ping_timeseries.png)

---

### 6.6 Multi-Chip Throughput

**`mc_d2h_throughput_heatmap.png`** — Heatmap of D2H peak throughput (GB/s) across every MMIO-mapped chip on the system. Rows = chips (identified by Tray ID / ASIC Location), columns = FIFO sizes (1 MB → 256 MB), fixed at 64 KB pages and 1 GB total transfer. Reveals per-chip performance variation across the tray.

![D2H Multi-Chip Throughput Heatmap — All Chips × FIFO Size](charts/mc_d2h_throughput_heatmap.png)

---

**`mc_d2h_throughput_vs_fifo.png`** — Line chart version of the multi-chip sweep: throughput vs FIFO size, one line per chip. Makes it easy to see which chips plateau earlier or higher than others.

![D2H Multi-Chip Throughput vs FIFO Size](charts/mc_d2h_throughput_vs_fifo.png)

---

**`h2d_mc_d2h_throughput_heatmap.png`** — Heatmap of H2D (DEVICE\_PULL) peak throughput (GB/s) across every chip on the system. Fixed at 256 KB pages across FIFO sizes 256 KB, 512 KB, 1 MB. Directly comparable to the D2H heatmap above — the throughput gap between high-bandwidth and low-bandwidth chips is visible in both directions.

![H2D Multi-Chip Throughput Heatmap — All Chips × FIFO Size](charts/h2d_mc_d2h_throughput_heatmap.png)

---

**`h2d_mc_d2h_throughput_vs_fifo.png`** — Line chart version of the H2D multi-chip sweep: DEVICE\_PULL throughput vs FIFO size, one line per chip.

![H2D Multi-Chip Throughput vs FIFO Size](charts/h2d_mc_d2h_throughput_vs_fifo.png)

---

**`h2d_mc_d2h_throughput_bar.png`** — Grouped bar chart of the H2D multi-chip sweep: one group per FIFO size, one bar per chip. An alternative view that makes magnitude differences between chips easier to compare at a glance.

![H2D Multi-Chip Throughput Bar Chart — All Chips × FIFO Size](charts/h2d_mc_d2h_throughput_bar.png)

---

## 7. Latency Methodology

### 7.1 What "round-trip latency" means here

**Round-trip latency** is the time from when the sender begins writing one page of data until it receives the receiver's acknowledgement that the page has been consumed. It includes:

- Time to write data across PCIe into the FIFO
- Time for the receiver to detect the data, copy it into its working buffer
- Time for the acknowledgement to travel back across PCIe

For **H2D**, the round-trip is: host calls `write(data, 1 page)` → device consumes page → device calls `socket_notify_sender` → host's `barrier()` returns.

For **D2H**, the round-trip is measured entirely on the device: device calls `socket_reserve_pages`, writes one page across PCIe, calls `socket_notify_receiver` → host reads and acks → device's `socket_barrier()` detects the ack.

### 7.2 Measurement approach — device-side cycle counters

All latency measurements use **device-side cycle counters** (`get_timestamp()` on the AI core), not host-side wall-clock timers. This eliminates host OS scheduling jitter, which can be hundreds of microseconds, and gives sub-microsecond resolution.

Per-iteration cycle deltas are written into a dedicated L1 measurement buffer — one `uint64_t` per iteration — and read back by the host after the kernel finishes:

```cpp
// Inside the device kernel (e.g. pcie_socket_ping.cpp)
for (uint32_t i = 0; i < num_iterations; i++) {
    uint64_t start_timestamp = get_timestamp();

    // ... send/receive one page ...

    uint64_t end_timestamp = get_timestamp();
    *reinterpret_cast<volatile uint64_t*>(measurement_buffer_addr + i * sizeof(uint64_t))
        = end_timestamp - start_timestamp;
}
```

The measurement buffer is sized as `num_iterations × sizeof(uint64_t)` and is read back via `cluster.read_core()` after `EnqueueMeshWorkload` completes.

To convert cycles to microseconds (Blackhole AI clock = 1.35 GHz):

```
Latency [µs] = cycles / 1350.0
```

### 7.3 Warmup

Both device kernels and host-side loops execute **5 warmup iterations** before timed iterations begin. Warmup flushes:

- Cold L1/L2 cache lines (the socket config, FIFO pointers, measurement buffer)
- PCIe TLB translation caches
- Host-side kernel scheduling effects

Without warmup, the first few iterations show latencies 2–5× higher than steady-state, making averages misleading.

**Host and device must agree on the warmup count.** The constant `WARMUP_ITERS = 5` appears in both the C++ kernel source and the corresponding host benchmark function. If they diverge, the host reads or writes the wrong number of pages and the kernel hangs waiting.

### 7.4 Reporting statistics

100 timed iterations are collected per configuration. The following statistics are reported:

| Statistic | Description |
|-----------|-------------|
| `avg_us` | Mean latency across all 100 iterations |
| `min_us` | Fastest observed round-trip |
| `max_us` | Slowest observed round-trip |
| `p50_us` | Median (50th percentile) |
| `p99_us` | 99th percentile — captures tail latency |
| `avg_cycles` | Mean in raw clock cycles |
| `min_cycles`, `max_cycles` | Range in cycles |

CSV output is emitted to stdout, one row per (page\_size, fifo\_size) configuration. The Python script `analyze_d2h_throughput.py` ingests this CSV and generates charts.

---

## 8. Throughput Methodology

### 8.1 What "throughput" means here

**Steady-state throughput** measures how fast the sender can push data when running continuously — i.e., how long it takes for successive pages to arrive at the receiver when both sides are pipelined. This is the metric most relevant to training workloads, where large tensors are streamed in bulk.

The measurement does **not** issue a barrier after each page. Instead, the sender writes all pages across all iterations with a single barrier at the end. This allows the sender and receiver pipelines to overlap, capturing the true sustained rate.

### 8.2 Measurement approach

Throughput is measured by the **device kernel** using a single pair of timestamps bracketing the entire transfer:

```cpp
// Inside pcie_socket_sender.cpp (D2H throughput)
uint64_t start_timestamp = get_timestamp();

for (uint32_t i = 0; i < num_iterations; i++) {
    for each page in data_size:
        socket_reserve_pages(...)
        // NOC write data to host FIFO
        socket_push_pages(...)
        socket_notify_receiver(...)
}
socket_barrier(sender_socket);  // wait for host to ack all data

uint64_t end_timestamp = get_timestamp();
*reinterpret_cast<volatile uint64_t*>(measurement_buffer_addr)
    = end_timestamp - start_timestamp;  // single uint64_t
```

From the total cycle count, per-page statistics are derived on the host:

```
total_pages = (data_size / page_size) × num_iterations
avg_cycles_per_page = total_cycles / total_pages
avg_us_per_page = avg_cycles_per_page / 1350.0
throughput_gbps = page_size_bytes / (avg_us_per_page × 1000)
```

Note: `throughput_gbps` here is expressed in **GB/s** (gigabytes per second, not gigabits), computed as bytes-per-page divided by microseconds-per-page.


## 10. Benchmark Tests (one paragraph each)

### D2HSocketThroughputBenchmark
The baseline D2H throughput test. The device kernel (`pcie_socket_sender.cpp`) writes `data_size` bytes per iteration using chunked NOC writes across PCIe into a host-pinned FIFO. A single pair of device-side timestamps brackets the full multi-iteration run; the host computes average per-page cycles and GB/s. Sweeps page sizes up to 256 KB and FIFO sizes up to 512 MB.

### D2HSocketLatencyBenchmark
Measures per-iteration round-trip latency on the D2H path with actual data DMA. Uses `pcie_socket_data_ping.cpp` on the device: each iteration sends one page to the host, then calls `socket_barrier` to wait for the host's acknowledgement. Five warmup iterations precede 100 timed iterations; each timed delta is stored in the L1 measurement buffer and read back for percentile reporting. Sweeps page sizes and a subset of FIFO sizes chosen to cover latency-sensitive operating points (1 KB to 512 MB FIFO).

### D2HSocketPingBenchmark
Measures **pure signalling overhead** on the D2H path — no data DMA occurs. Uses `pcie_socket_ping.cpp`: the device calls `socket_reserve_pages` / `socket_push_pages` / `socket_notify_receiver` / `socket_barrier` with no actual payload write. The host calls `output_socket.read()` to consume the page slot and send the ack. This isolates the flow-control protocol overhead from the data transfer cost. The first config also dumps raw per-iteration data to `tests/tt_metal/distributed/ping_iterations.csv` for jitter analysis.

### D2HSocketMultiChipMaxThroughputBenchmark
Sweeps **every MMIO-mapped chip** on the system (identified via `PhysicalSystemDescriptor` + tray/ASIC location metadata) and measures D2H throughput at 64 KB pages (the empirically best page size for throughput) across five FIFO sizes (1 MB, 4 MB, 16 MB, 64 MB, 256 MB). Produces a CSV with tray ID, ASIC location, and mesh coordinate columns so per-chip variation across the tray can be compared. Total data transferred per configuration: 1 GB.

### H2DSocketMultiChipMaxThroughputBenchmark
Sweeps **every chip** on the system (all 32 MMIO-mapped chips in a Blackhole Galaxy) and measures H2D throughput using **DEVICE\_PULL** at 256 KB pages — the empirically highest-throughput page size for this mode — across three FIFO sizes: 256 KB, 512 KB, and 1 MB. Produces a CSV with tray ID, ASIC location, and mesh coordinate columns so per-chip variation across the entire system can be compared. Total data transferred per configuration: 1 GB. Analogous to `D2HSocketMultiChipMaxThroughputBenchmark` on the H2D path.

### H2DSocketThroughputBenchmark
Measures H2D steady-state throughput for both `HOST_PUSH` and `DEVICE_PULL` modes in a single test. For HOST\_PUSH, the host issues a TLB write per page and the device kernel (`h2d_throughput_host_push.cpp`) timestamps the full receive loop. For DEVICE\_PULL, the host writes to pinned memory and the device kernel (`h2d_throughput_device_pull.cpp`) issues chunked NOC reads and timestamps the full loop. Both kernels report a single aggregate cycle count; the host derives per-page GB/s. Sweeps FIFO sizes up to 1 MB, page sizes up to 256 KB.

### H2DSocketLatencyBenchmark
Measures per-iteration round-trip latency on the H2D path for both `HOST_PUSH` and `DEVICE_PULL`. Uses `h2d_socket_data_ping_host_push.cpp` and `h2d_socket_data_ping_device_pull.cpp` respectively. Both kernels follow the same 5-warmup + 100-timed-iteration pattern with per-iteration L1 measurement buffers.

**What the device timer captures:** Each iteration records `start = get_timestamp()` *before* calling `socket_wait_for_pages`, then `end = get_timestamp()` after `socket_notify_sender` completes. Since the device starts the timer before the data has arrived, the measured cycles include: (a) spin-wait for the host's write to propagate over PCIe, (b) device-side copy to local buffer, and (c) the acknowledgement NOC write back to host pinned memory. This covers the full round-trip as seen from the device. On the host side, each iteration calls `input_socket.write()` then `input_socket.barrier()` (which waits for the ack); the host does not independently time iterations — device cycle counters are the sole measurement source.

### H2DSocketPingBenchmark
Measures **pure signalling overhead** on the H2D path for both modes. Uses `h2d_socket_ping.cpp` on the device (no DMA in device kernel). The host issues `write + barrier` per iteration (5 warmup + 100 timed). Each iteration's cycle delta is stored per-iteration and also dumped to `h2d_ping_iterations_HOST_PUSH.csv` / `h2d_ping_iterations_DEVICE_PULL.csv` for jitter analysis. Compares mode overhead directly since the kernel path is identical for both modes.

---

## 11. Key Formulas and Constants

| Constant / Formula | Value / Expression |
|---|---|
| AI clock (Blackhole) | **1.35 GHz** |
| Cycles → microseconds | `latency_us = cycles / 1350.0` |
| Throughput from per-page latency | `gbps = page_size_bytes / (avg_us_per_page × 1000)` |
| L1 data budget (max data\_size) | **1,400,000 bytes (~1.4 MB)** — governs `pages_per_iter` for large pages |
| Pages per iteration | **Always 1** — `kPagesPerIteration = 1` is hard-coded; one page is sent/received per benchmark iteration |
| Warmup iterations | **5** (both host and device must match) |
| Timed iterations (latency) | **100** |
| NOC burst constraint | `NOC_MAX_BURST_SIZE` — pages larger than this require chunked NOC transfers |
| Target chip (single-chip tests) | Tray **1**, ASIC Location **6** |

---

## 12. Interpreting Results

### Throughput vs. FIFO size

Throughput rises as FIFO size grows and then **plateaus**. The plateau begins when the FIFO is large enough that the sender is never stalled waiting for receiver acknowledgements. Before the plateau, the sender is back-pressured after every page (or small batch), and the throughput equals approximately `page_size / round_trip_latency`. After the plateau, throughput is limited by the PCIe bandwidth ceiling.

See **§6.1** (`d2h_tp_vs_fifo.png`) and **§6.3** (`h2d_tp_vs_fifo.png`) to observe where the plateau occurs for each page size and mode.

### Throughput vs. page size

Very small pages (64–256 B) have very low throughput because the per-page fixed overhead (NOC command setup, PCIe transaction framing, `bytes_sent` notification write) dominates over the data transfer time. Throughput rises roughly linearly with page size until it saturates the PCIe link bandwidth, typically around 16–64 KB pages.

See **§6.1** (`d2h_throughput.png`) and **§6.3** (`h2d_throughput.png`) for the page-size saturation curves.

### Latency vs. page size

Latency grows with page size because more data must traverse PCIe. For small pages the dominant cost is the protocol overhead (roughly a fixed number of NOC round-trips), not the data volume. For large pages the DMA time dominates. The log-log scale in **§6.2** (`d2h_latency.png`) and **§6.4** (`h2d_latency.png`) makes both regimes visible as distinct slopes.

### HOST\_PUSH vs. DEVICE\_PULL (H2D)

- **HOST\_PUSH** generally has lower latency because the host can write directly into device L1 with a single TLB write, avoiding the device issuing a separate NOC read.
- **DEVICE\_PULL** can achieve higher sustainable throughput in CPU-bottlenecked scenarios because the host only needs to update `bytes_sent` (a 4-byte write), while the device handles the bulk DMA itself. This also frees the host CPU for other work.
- See **§6.4** (`h2d_latency.png`) for the direct latency comparison and **§6.3** (`h2d_throughput.png`) for the throughput comparison.

### Tail latency (p99 vs. avg)

Large gaps between `avg_us` and `p99_us` indicate interference from the OS scheduler, PCIe power management, or NUMA effects. The warmup is designed to minimise this for early iterations, but OS preemption can still spike individual iterations. Training teams should budget for p99 latency, not average, when sizing timeout windows or synchronisation barriers.

The per-iteration ping plots (**§6.5**) expose this jitter directly — any iteration that spikes significantly above the median is a scheduling artefact, not a hardware limit.

### Per-chip throughput variation

As described in **§1.1**, the tray contains two fundamentally different chip classes:

| Chip class | PCIe link | D2H ceiling | H2D ceiling |
|------------|-----------|-------------|-------------|
| High-bandwidth (4 chips, e.g. ASIC 6) | Gen 4 ×8 | **~15.1 GB/s** | **~11.5 GB/s** (DEVICE\_PULL) |
| Low-bandwidth (28 chips, e.g. ASIC 1) | Gen 1 ×1 | **~0.21 GB/s** | **~0.21 GB/s** (DEVICE\_PULL) |

The 70× D2H throughput gap between ASIC 6 and ASIC 1 is entirely explained by the PCIe link difference — it does not improve with larger pages or larger FIFOs.

Within the 4 high-bandwidth chips there is also chip-to-chip variation (a few percent) driven by NUMA topology and PCIe root complex distance. See **§6.6** (`mc_d2h_throughput_heatmap.png`) to identify the highest-throughput chip in your specific system before pinning latency-sensitive jobs.

---

## 13. Running the Benchmarks

All tests require a system with vIOMMU enabled. They will `GTEST_SKIP` automatically on unsupported systems via the `GetMemoryPinningParameters` check.

Build the test binary:
```bash
./build_metal.sh
```

Run a specific benchmark (e.g., D2H latency):
```bash
./build/test/tt_metal/distributed/test_hd_sockets \
    --gtest_filter="HDSocketFixture.D2HSocketLatencyBenchmark" \
    2>&1 | tee d2h_latency_results.csv
```

Analyse and plot results:
```bash
# Throughput chart
python3 tests/tt_metal/distributed/analyze_d2h_throughput.py \
    --throughput d2h_throughput_results.csv

# Latency chart
python3 tests/tt_metal/distributed/analyze_d2h_throughput.py \
    --latency d2h_latency_results.csv

# Multi-chip throughput chart
python3 tests/tt_metal/distributed/analyze_d2h_throughput.py \
    --multichip multichip_bench.log
```

Output CSV columns (throughput):
```
page_size, socket_fifo_size, [h2d_mode,] total_data, data_size,
pages_per_iter, num_iterations, total_pages,
avg_per_page_us, avg_per_page_cycles, throughput_gbps, device_coord
```

Output CSV columns (latency):
```
page_size, socket_fifo_size, [h2d_mode,] num_iterations,
avg_us, min_us, max_us, p50_us, p99_us,
avg_cycles, min_cycles, max_cycles, device_coord
```

> **Note on CSV parsing:** `device_coord` (a `MeshCoordinate`) contains an embedded comma, e.g. `(0, 1)`. The Python analysis script uses a custom row parser (`_read_split_rows`) to handle this correctly; do not attempt to parse the output with a naive `split(",")`.

---

## 14. Appendix — Detailed Parameter-Space Charts

These charts are generated by the same script but show the full (page\_size × FIFO\_size × total\_data) parameter space. They are useful for deep dives but not necessary to understand the headline results.

---

### A.1 D2H Full Parameter-Space Heatmap Grid

**`d2h_tp_heatmap_grid.png`** — A grid of heatmaps, one subplot per total-data size. X-axis = FIFO size, Y-axis = page size, cell colour = throughput in GB/s. Annotated with numeric values. The most exhaustive view of the D2H throughput space.

![D2H Throughput Heatmap Grid — Full Parameter Space](charts/d2h_tp_heatmap_grid.png)

---

### A.2 D2H Latency Breakdown

**`d2h_latency_breakdown.png`** — Side-by-side bar charts: p50 latency (left panel) and max latency (right panel), grouped by page size with one bar cluster per FIFO size. Useful for seeing whether outlier max values are proportional to p50 or represent large isolated spikes.

![D2H Latency Breakdown — p50 and Max by Page Size](charts/d2h_latency_breakdown.png)

---

### A.3 H2D Latency — Per-FIFO Breakdown

**`h2d_latency_breakdown.png`** — Two panels (one per H2D mode), showing p50 latency vs page size with one line per FIFO size. Lets you verify that FIFO size has little effect on latency (which it should for single-page round-trips), or detect unexpected FIFO-size sensitivity.

![H2D Latency Breakdown — All FIFO Sizes, HOST_PUSH vs DEVICE_PULL](charts/h2d_latency_breakdown.png)

---

### A.4 H2D Throughput at Maximum FIFO

**`h2d_tp_at_max_fifo.png`** — Two panels (one per H2D mode), throughput vs page size, lines = total-data sizes, fixed at the maximum FIFO. Complements `h2d_throughput.png` by showing how throughput scales with the amount of data moved for each mode independently.

![H2D Throughput at Max FIFO — All Total-Data Sizes](charts/h2d_tp_at_max_fifo.png)

---

### A.5 Multi-Chip Throughput Bar Chart

**`mc_d2h_throughput_bar.png`** — Grouped bar chart of the multi-chip sweep: one group per FIFO size, one bar per chip. An alternative view to the heatmap (§6.6) that makes magnitude differences between chips easier to compare at a glance.

![D2H Multi-Chip Throughput Bar Chart — All Chips × FIFO Size](charts/mc_d2h_throughput_bar.png)
