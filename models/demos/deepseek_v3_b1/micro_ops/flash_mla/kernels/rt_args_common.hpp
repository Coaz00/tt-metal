// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <tuple>

inline uint32_t nearest_n(uint32_t x, uint32_t n) { return ((x + n - 1) / n) * n; }

// Returns whether this SP device has any work for the given cur_pos.
// Tokens are assigned in device_chunk_size groups round-robin across SP devices:
//   [0, dcs) -> dev 0, [dcs, 2*dcs) -> dev 1, ..., [num_sp*dcs, (num_sp+1)*dcs) -> dev 0, ...
// Device d has work iff the global valid token count exceeds d * device_chunk_size.
inline bool device_has_work(uint32_t cur_pos, uint32_t sp_device_idx, uint32_t device_chunk_size) {
    return cur_pos >= sp_device_idx * device_chunk_size;
}

inline std::tuple<uint32_t, uint32_t, uint32_t> get_runtime_args(
    int cur_pos,
    int cur_batch,
    int core_num,
    int num_cores_per_batch,
    uint32_t k_chunk_size,
    uint32_t device_chunk_size = 0,
    uint32_t sp_device_idx = 0,
    uint32_t num_sp_devices = 1) {
    // Global valid sequence length (rounded up to k_chunk_size)
    uint32_t valid_seq_len = nearest_n(cur_pos + 1, k_chunk_size);
    uint32_t num_chunks_value = valid_seq_len / k_chunk_size;

    // When SP is active, compute local chunk count for this device.
    // Device-chunk groups of (device_chunk_size / k_chunk_size) chunks are distributed
    // round-robin across SP devices:
    //   group 0 -> dev 0, group 1 -> dev 1, ..., group num_sp -> dev 0, ...
    if (num_sp_devices > 1) {
        uint32_t chunks_per_dc = device_chunk_size / k_chunk_size;
        uint32_t total_groups = num_chunks_value / chunks_per_dc;
        uint32_t remaining_chunks = num_chunks_value % chunks_per_dc;

        uint32_t groups_for_device = total_groups / num_sp_devices;
        uint32_t extra_group_cutoff = total_groups % num_sp_devices;
        if (sp_device_idx < extra_group_cutoff) {
            groups_for_device += 1;
        }

        num_chunks_value = groups_for_device * chunks_per_dc;

        if (remaining_chunks > 0 && sp_device_idx == extra_group_cutoff) {
            num_chunks_value += remaining_chunks;
        }
    }

    uint32_t k_chunk_start = 0;
    uint32_t k_chunk_end = 0;

    // Strided chunk distribution: core N gets chunks N, N+num_cores, N+2*num_cores, ...
    // This ensures each core reads from the same DRAM bank (round-robin sharding)
    // E.g., with 8 cores and 16 chunks:
    //   core 0 gets chunks 0, 8 (both on bank 1)
    //   core 1 gets chunks 1, 9 (both on bank 3)
    //   etc.
    if (num_cores_per_batch > int(num_chunks_value)) {
        // More cores than chunks: each active core gets 1 chunk
        int chunks_per_core = (core_num < int(num_chunks_value)) ? 1 : 0;
        k_chunk_start = core_num;
        k_chunk_end = k_chunk_start + chunks_per_core;
    } else {
        // More chunks than cores: strided distribution
        // k_chunk_start = core_num (first chunk index)
        // k_chunk_end = first chunk + (num_chunks - 1) * stride + 1
        // Kernel iterates: for (k = start; k < end; k += stride)
        // where stride = num_cores_per_batch
        int chunks_per_core = num_chunks_value / num_cores_per_batch;
        int residuals = num_chunks_value % num_cores_per_batch;
        int num_chunks_for_core = chunks_per_core + (core_num < residuals ? 1 : 0);

        k_chunk_start = core_num;
        k_chunk_end =
            k_chunk_start + (num_chunks_for_core > 0 ? (num_chunks_for_core - 1) * num_cores_per_batch + 1 : 0);
    }

    return {num_chunks_value, k_chunk_start, k_chunk_end};
}
