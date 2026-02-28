# Lightweight Mask for Ring Joint SDPA

## Problem

The ring joint SDPA allocates a full `Sq_chunk_t * Sk_chunk_t` mask matrix in circular buffer `c_3` using `Bfp4_b` format. This consumes significant L1 memory, especially for large chunk sizes.

### How the old mask works

The writer kernel calls `generate_noncausal_padded_mask` once per Q chunk. This function walks every tile position in the `Sq_chunk_t * Sk_chunk_t` matrix and fills each tile with one of three patterns:

1. **Zero tile** — valid K position, no masking needed. `QK + 0 = QK` (no-op).
2. **`-inf` tile** — fully padded K position. `QK + (-inf) = -inf`, which softmax drives to zero.
3. **Vertical partial tile** — boundary tile where the padding edge falls _inside_ the tile. Left columns are zero, right columns are `-inf`.

The compute kernel then calls `add_block_inplace(cb_qk_im, cb_mask_in, qk_chunk_tiles)`, which iterates over **all** `Sq_chunk_t * Sk_chunk_t` tiles and adds them element-wise to QK. This means every tile is touched — even the zero tiles that have no effect.

### Why this is wasteful

Consider `Sq_chunk_t=4, Sk_chunk_t=8` with 5 valid K tile columns and 3 padded columns:

```
QK matrix (4 rows x 8 cols of tiles):

  K0   K1   K2   K3   K4  | K5   K6   K7
┌────┬────┬────┬────┬────┐┌────┬────┬────┐
│ qk │ qk │ qk │ qk │ qk ││ qk │ qk │ qk │  Q0
├────┼────┼────┼────┼────┤├────┼────┼────┤
│ qk │ qk │ qk │ qk │ qk ││ qk │ qk │ qk │  Q1
├────┼────┼────┼────┼────┤├────┼────┼────┤
│ qk │ qk │ qk │ qk │ qk ││ qk │ qk │ qk │  Q2
├────┼────┼────┼────┼────┤├────┼────┼────┤
│ qk │ qk │ qk │ qk │ qk ││ qk │ qk │ qk │  Q3
└────┴────┴────┴────┴────┘└────┴────┴────┘
         valid                  padded

Old mask matrix (same shape, stored entirely in CB c_3):

  K0   K1   K2   K3   K4    K5   K6   K7
┌────┬────┬────┬────┬────┬────┬────┬────┐
│  0 │  0 │  0 │  0 │  0 │-inf│-inf│-inf│  Q0
├────┼────┼────┼────┼────┼────┼────┼────┤
│  0 │  0 │  0 │  0 │  0 │-inf│-inf│-inf│  Q1
├────┼────┼────┼────┼────┼────┼────┼────┤
│  0 │  0 │  0 │  0 │  0 │-inf│-inf│-inf│  Q2
├────┼────┼────┼────┼────┼────┼────┼────┤
│  0 │  0 │  0 │  0 │  0 │-inf│-inf│-inf│  Q3
└────┴────┴────┴────┴────┴────┴────┴────┘

add_block_inplace touches all 32 tiles.
20 of them are zero — adding them is pure waste.
```

**L1 cost**: 32 tiles in `Bfp4_b` format = 32 * 288 bytes = 9,216 bytes.

## Solution

Replace the full mask matrix with a **single permanently-fronted `-inf` tile** in `Float16_b` format.

The compute kernel only L1-accumulates this single `-inf` tile onto the fully padded K-column positions. Valid positions (zero tiles) are skipped entirely.

```
Lightweight approach (same example):

CB c_3 holds just 1 tile:
┌────┐
│-inf│  (permanently fronted, never popped)
└────┘

Compute only touches the 12 padded tiles (K5..K7 x Q0..Q3):

  K0   K1   K2   K3   K4    K5   K6   K7
┌────┬────┬────┬────┬────┬────┬────┬────┐
│skip│skip│skip│skip│skip│+inf│+inf│+inf│  Q0
├────┼────┼────┼────┼────┼────┼────┼────┤
│skip│skip│skip│skip│skip│+inf│+inf│+inf│  Q1
├────┼────┼────┼────┼────┼────┼────┼────┤
│skip│skip│skip│skip│skip│+inf│+inf│+inf│  Q2
├────┼────┼────┼────┼────┼────┼────┼────┤
│skip│skip│skip│skip│skip│+inf│+inf│+inf│  Q3
└────┴────┴────┴────┴────┴────┴────┴────┘
                          ^^^^^^^^^^^^^^^^
                    Only these 12 tiles are written via
                    L1 accumulation from the single -inf tile
```

**L1 cost**: 1 tile in `Float16_b` = 1 * 1,024 bytes = 1,024 bytes. **(~9x reduction)**

| | Before | After |
|---|---|---|
| CB `c_3` size | `Sq_chunk_t * Sk_chunk_t` tiles (`Bfp4_b`) | 1 tile (`Float16_b`) |
| Writer work | `generate_noncausal_padded_mask` per Q chunk | Single `-inf` tile fill at startup |
| Compute work | `add_block_inplace` over all tiles | `apply_padded_mask_lightweight_runtime` over padded tiles only |

## Detailed Code Changes

### 1. `ring_joint_sdpa_program_factory.cpp` (host)

**Constraint computation** — determines whether lightweight mask is safe:

```cpp
const bool use_lightweight_mask =
    (args.logical_n % tt::constants::TILE_HEIGHT == 0) &&
    (L % tt::constants::TILE_HEIGHT == 0);
```

**Compile-time args** — the flag is passed to both kernels:
- Writer: arg index 20 (`(std::uint32_t)use_lightweight_mask`)
- Compute: arg index 31 (`(std::uint32_t)use_lightweight_mask`)

**CB `c_3` sizing** — conditional on the flag:

```cpp
const uint32_t actual_mask_tiles = use_lightweight_mask ? 1 : mask_tiles;
const tt::DataFormat actual_mask_df = use_lightweight_mask ? tt::DataFormat::Float16_b : mask_df;
```

When `use_lightweight_mask` is false, this falls back to the original `mask_tiles` (= `Sq_chunk_t * Sk_chunk_t`) in `Bfp4_b`.

### 2. `ring_joint_writer.cpp` (writer kernel)

**Reads the flag** at arg 20:

```cpp
constexpr bool use_lightweight_mask = get_compile_time_arg_val(20) == 1;
```

**`TensorAccessorArgs` offset** bumped from 20 to 21 because the new arg was inserted before it:

```cpp
constexpr auto out_args = TensorAccessorArgs<21>();  // was 20
```

**Single `-inf` tile generation** — runs once at startup, the tile stays permanently fronted in CB `c_3` for the entire kernel lifetime:

```cpp
if constexpr (use_lightweight_mask) {
    const uint32_t mask_tile_size_bytes = get_tile_size(cb_mask_in);
    cb_reserve_back(cb_mask_in, 1);
    auto* ptr = reinterpret_cast<uint32_t*>(get_write_ptr(cb_mask_in));
    for (uint32_t i = 0; i < mask_tile_size_bytes / sizeof(uint32_t); i++) {
        ptr[i] = 0xFF80FF80;  // Two bfloat16 -inf values packed into one uint32
    }
    cb_push_back(cb_mask_in, 1);
}
```

`0xFF80` is `-inf` in bfloat16. Two are packed per `uint32_t` write. The tile is pushed but never popped — the compute kernel reads it repeatedly via index 0.

**`generate_mask` bypass** — the per-Q-chunk mask generation is wrapped in:

```cpp
if constexpr (!use_lightweight_mask) {
    generate_mask<...>(...);
}
```

This eliminates all per-chunk mask generation overhead.

### 3. `ring_joint_sdpa.cpp` (compute kernel)

**Reads the flag** at arg 31:

```cpp
constexpr bool use_lightweight_mask = get_compile_time_arg_val(31) == 1;
```

**Padded tile count computation** — the ring joint SDPA has 3 independent mask types. For each, we compute how many K tile columns in the boundary chunk are fully padded:

```cpp
uint32_t global_n_padded_tiles = 0;
uint32_t local_n_padded_tiles = 0;
constexpr uint32_t joint_n_padded_tiles =
    (Lt % Sk_chunk_t != 0) ? (Sk_chunk_t - (Lt % Sk_chunk_t)) : 0;

if constexpr (use_lightweight_mask) {
    if (ring_iter_needs_global_n_mask) {
        const uint32_t unpadded_in_chunk =
            global_n_within_ring_iter % (Sk_chunk_t * tt::constants::TILE_HEIGHT);
        const uint32_t valid_tiles =
            (unpadded_in_chunk + tt::constants::TILE_HEIGHT - 1) / tt::constants::TILE_HEIGHT;
        global_n_padded_tiles = Sk_chunk_t - valid_tiles;
    }
    if (local_n_needs_masking) {
        local_n_padded_tiles = Sk_chunk_t - (local_padded_Nt % Sk_chunk_t);
    }
}
```

Example: if `Sk_chunk_t=8` and `local_padded_Nt=13`, then `13 % 8 = 5` valid tile columns in the last chunk, so `local_n_padded_tiles = 8 - 5 = 3`.

These counts are passed to `sdpa_ring` → `sdpa_inner_loop`.

### 4. `compute_common.hpp`

#### New function: `apply_padded_mask_lightweight_runtime`

This replaces `add_block_inplace` for the lightweight path. Instead of adding a full mask matrix to QK, it L1-accumulates the single `-inf` tile onto only the padded positions:

```cpp
void apply_padded_mask_lightweight_runtime(
    uint32_t neginf_cb, uint32_t out_cb,
    uint32_t num_padded, uint32_t num_cols, uint32_t num_rows)
```

**How it works:**

1. `start = num_cols - num_padded` — first padded column index
2. Enables L1 accumulation mode (`llk_pack_reconfig_l1_acc(1)`) — `pack_tile<true>` will ADD the tile content into the destination CB instead of overwriting
3. For each Q row: iterates from `start` to `num_cols`, batching up to 8 tiles in DST registers at a time
4. For each batch: copies the single `-inf` tile (always index 0) into DST slots, then packs them back into `cb_qk_im` at the correct offsets with L1 accumulation
5. Disables L1 accumulation mode afterward

```
Example: num_cols=8, num_padded=3, num_rows=4

For each row, processes tiles at columns 5, 6, 7:
  Row 0: pack_tile<true> at offsets 5, 6, 7     (QK[0][5..7] += -inf)
  Row 1: pack_tile<true> at offsets 13, 14, 15   (QK[1][5..7] += -inf)
  Row 2: pack_tile<true> at offsets 21, 22, 23   (QK[2][5..7] += -inf)
  Row 3: pack_tile<true> at offsets 29, 30, 31   (QK[3][5..7] += -inf)

Only 12 tile operations instead of 32.
```

#### Modified mask application in `sdpa_inner_loop` (RING path)

The existing `apply_mask` logic determines _which_ chunk needs masking. Inside that block, the code now branches:

```cpp
if (use_lightweight_mask) {
    // Determine num_padded based on which mask type applies to this chunk
    uint32_t num_padded = 0;
    if (ring_iter_needs_global_n_mask && k_chunk == global_n_mask_chunk_id) {
        num_padded = global_n_padded_tiles;
    } else if (local_n_needs_masking && k_chunk == local_n_mask_chunk_id) {
        num_padded = local_n_padded_tiles;
    } else if (ring_iter_needs_joint_n_mask &&
               (k_chunk - num_local_k_chunks) == joint_n_mask_chunk_id) {
        num_padded = joint_n_padded_tiles;
    }
    if (num_padded > 0) {
        apply_padded_mask_lightweight_runtime(
            cb_mask_in, cb_qk_im, num_padded, Sk_chunk_t, Sq_chunk_t);
    }
} else {
    add_block_inplace(cb_qk_im, cb_mask_in, qk_chunk_tiles);
}
```

The key insight: only **one** mask type applies to any given K chunk (they are mutually exclusive per chunk), so we pick the right padded count and apply.

#### Default parameters on `sdpa_inner_loop` and `sdpa_ring`

Four new parameters with defaults so existing callers (standard SDPA, joint SDPA) are unaffected:

```cpp
const bool use_lightweight_mask = false,
const uint32_t global_n_padded_tiles = 0,
const uint32_t local_n_padded_tiles = 0,
const uint32_t joint_n_padded_tiles = 0
```

## Illustrative Example

### Setup

- `Sq_chunk_t = 2`, `Sk_chunk_t = 4` (Q chunk = 2 tile rows, K chunk = 4 tile cols)
- `logical_n = 320` (total valid K length = 320 elements = 10 tiles)
- `local_padded_N = 384` (padded to 384 = 12 tiles, `local_padded_Nt = 12`)
- Ring with 2 devices, each holding 12 K tiles → 3 K chunks of 4 tiles each
- Ring iter 0 processes device 0's KV (K tiles 0..11)

Since `logical_n = 320 = 10 tiles`, and ring iter 0 starts at tile 0, the global N boundary falls at tile 10 — inside K chunk 2 (tiles 8..11). Tiles 10 and 11 are padded.

### Old approach (full mask matrix)

**Writer** generates a `2 x 4` mask matrix (8 tiles) for the boundary K chunk:

```
K chunk 2 mask (Sq_chunk_t=2 x Sk_chunk_t=4):
┌────┬────┬────┬────┐
│  0 │  0 │-inf│-inf│  Q row 0
├────┼────┼────┼────┤
│  0 │  0 │-inf│-inf│  Q row 1
└────┴────┴────┴────┘
  K8   K9   K10  K11
```

CB `c_3` stores 8 tiles in `Bfp4_b`.

**Compute** calls `add_block_inplace(cb_qk_im, cb_mask_in, 8)` — processes all 8 tiles. The 4 zero tiles add nothing.

### New approach (lightweight mask)

**Writer** generates one `-inf` tile at startup. No per-chunk work.

CB `c_3` stores 1 tile in `Float16_b`.

**Compute** knows `global_n_padded_tiles = 2` (tiles K10, K11 are padded).

Calls `apply_padded_mask_lightweight_runtime(cb_mask_in, cb_qk_im, 2, 4, 2)`:
- `start = 4 - 2 = 2` → begins at column index 2
- Row 0: accumulates `-inf` at QK[0][2] and QK[0][3]
- Row 1: accumulates `-inf` at QK[1][2] and QK[1][3]

**4 tile operations instead of 8. CB uses 1,024 bytes instead of 2,304 bytes.**

For K chunks 0 and 1, no mask is applied at all — those chunks are fully valid, so `apply_mask` is false and the entire masking step is skipped.

## Constraints

The lightweight mask is **automatically enabled** only when all padding boundaries are tile-aligned:

```
use_lightweight_mask = (logical_n % TILE_HEIGHT == 0) && (L % TILE_HEIGHT == 0)
```

- **`logical_n % TILE_HEIGHT == 0`**: Ensures the global N boundary does not produce a partial tile (a tile where the left half is valid and the right half is padded).
- **`L % TILE_HEIGHT == 0`**: Ensures the joint L boundary does not produce a partial tile.
- **`local_padded_N`**: Always tile-aligned by construction, so no constraint is needed.

When either condition is not met, the code falls back to the full mask matrix path (`Bfp4_b`, `generate_mask` + `add_block_inplace`).

## Limitations

The lightweight approach only handles **fully padded tiles** — it can set an entire tile to `-inf` but cannot mask individual columns within a tile. If the padding boundary falls inside a tile (partial tile), the old `generate_noncausal_padded_mask` path is needed because it generates a special "vertical" tile with per-column masking. This is why the constraints above gate activation.
