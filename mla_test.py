import ttnn
import torch
import numpy as np


device = ttnn.open_mesh_device(ttnn.MeshShape([1, 1]))

# Using scaled_dot_product_attention_decode with separate Q, K, V tensors
# V is now a completely separate tensor, not extracted from K
batch_size, num_heads, seq_len = 1, 16, 1024
head_dim = 192  # Q and K dimension
v_dim = 128  # V dimension (can be different from K dimension)

# Q tensor for decode: (batch, num_heads, 1, head_dim) - sequence length is 1 for decode
q = torch.randn(batch_size, num_heads, 1, head_dim).float()

# K tensor: (batch, num_heads, seq_len, head_dim)
k = torch.randn(batch_size, num_heads, seq_len, head_dim).float()

# V tensor: (batch, num_heads, seq_len, v_dim) - separate tensor with different dimension
v = torch.randn(batch_size, num_heads, seq_len, v_dim).float()

# Convert to TTNN tensors
# Q needs to be permuted: (B, H, S, D) -> (S, B, H, D) for TTNN
tt_q = ttnn.from_torch(
    q.permute(2, 0, 1, 3),  # (1, B, H, D)
    device=device,
    dtype=ttnn.bfloat16,
    layout=ttnn.TILE_LAYOUT,
    memory_config=ttnn.DRAM_MEMORY_CONFIG,
)

tt_k = ttnn.from_torch(
    k,
    device=device,
    dtype=ttnn.bfloat16,
    layout=ttnn.TILE_LAYOUT,
    memory_config=ttnn.DRAM_MEMORY_CONFIG,
)

tt_v = ttnn.from_torch(
    v,
    device=device,
    dtype=ttnn.bfloat16,
    layout=ttnn.TILE_LAYOUT,
    memory_config=ttnn.DRAM_MEMORY_CONFIG,
)

# Scale factor
scale = head_dim**-0.5

# Current position tensor (for decode)
cur_pos = [seq_len // 2]  # Start position
tt_cur_pos = ttnn.from_torch(
    torch.tensor(cur_pos, dtype=torch.int32),
    device=device,
    dtype=ttnn.int32,
)

# Create program config
sdpa_program_config = ttnn.SDPAProgramConfig(
    compute_with_storage_grid_size=device.compute_with_storage_grid_size(),
    q_chunk_size=0,  # Not used in decode
    k_chunk_size=128,
    exp_approx_mode=False,
)

# Compute kernel config
compute_kernel_config = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4,
    math_approx_mode=False,
    fp32_dest_acc_en=False,
    packer_l1_acc=False,
)

# Call scaled_dot_product_attention_decode with separate Q, K, V tensors
output = ttnn.transformer.flash_multi_latent_attention_decode(
    tt_q,
    tt_k,
    tt_v,  # Separate V tensor
    head_dim_v=v_dim,
    is_causal=True,
    cur_pos_tensor=tt_cur_pos,
    scale=scale,
    memory_config=ttnn.DRAM_MEMORY_CONFIG,
)
print(output)
print(f"Output shape: {output.shape}")
print("Success!")
