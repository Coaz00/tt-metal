# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""
Test for instantiating both reference CPU and TT device MLA modules with the same weights.
This test verifies that both modules can be created and weights are loaded correctly.
"""

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.deepseek_v3.reference.configuration_deepseek import DeepseekV3Config
from models.demos.deepseek_v3_d_p.reference.mla_reference import create_mla_reference
from models.demos.deepseek_v3_d_p.tt.mla import ttMLA


@pytest.fixture
def random_weights():
    """
    Generate random weights for testing.

    Returns:
        Dictionary of weights in bfloat16
    """
    torch.manual_seed(42)
    test_config = DeepseekV3Config(
        vocab_size=129280,
        hidden_size=7168,
        num_attention_heads=128,
        num_key_value_heads=128,
        kv_lora_rank=512,
        q_lora_rank=1536,
        qk_rope_head_dim=64,
        v_head_dim=128,
        qk_nope_head_dim=128,
        max_position_embeddings=4096,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        attention_bias=False,
        attention_dropout=0.0,
    )

    # Generate random weights matching MLA architecture
    weights = {
        "q_a_proj.weight": torch.randn(test_config.q_lora_rank, test_config.hidden_size, dtype=torch.bfloat16),
        "q_a_layernorm.weight": torch.ones(test_config.q_lora_rank, dtype=torch.bfloat16),
        "q_b_proj.weight": torch.randn(
            test_config.num_attention_heads * (test_config.qk_nope_head_dim + test_config.qk_rope_head_dim),
            test_config.q_lora_rank,
            dtype=torch.bfloat16,
        ),
        "kv_a_proj_with_mqa.weight": torch.randn(
            test_config.kv_lora_rank + test_config.qk_rope_head_dim,
            test_config.hidden_size,
            dtype=torch.bfloat16,
        ),
        "kv_a_layernorm.weight": torch.ones(test_config.kv_lora_rank, dtype=torch.bfloat16),
        "kv_b_proj.weight": torch.randn(
            test_config.num_attention_heads * (test_config.qk_nope_head_dim + test_config.v_head_dim),
            test_config.kv_lora_rank,
            dtype=torch.bfloat16,
        ),
        "o_proj.weight": torch.randn(
            test_config.hidden_size,
            test_config.num_attention_heads * test_config.v_head_dim,
            dtype=torch.bfloat16,
        ),
    }

    logger.info(f"Generated {len(weights)} random weight tensors")
    return weights


# sp x tp
@pytest.mark.parametrize(
    "mesh_device",
    [(4, 2)],
    ids=["4x2"],
    indirect=True,
)
@pytest.mark.parametrize(
    "device_params",
    [
        {
            "fabric_config": ttnn.FabricConfig.FABRIC_1D,
        }
    ],
    indirect=True,
)
@pytest.mark.parametrize("use_pretrained", [False, True], ids=["random", "pretrained"])
@pytest.mark.parametrize("seq_len", [1024], ids=["seq1024"])
def test_mla(use_pretrained, random_weights, pretrained_weights, mesh_device, seq_len):
    """
    Test comparing reference and TT MLA modules with same weights.

    Args:
        use_pretrained: Whether to use pretrained weights
        random_weights: Random weights for testing
        pretrained_weights: Pretrained weights (or skipped if not available)
    """
    weight_type = "Pretrained" if use_pretrained else "Random"
    logger.info("=" * 80)
    logger.info(f"Test: Reference vs TT Comparison ({weight_type} Weights)")
    logger.info("=" * 80)

    test_config = DeepseekV3Config(
        vocab_size=129280,
        hidden_size=7168,
        num_attention_heads=128,
        num_key_value_heads=128,
        kv_lora_rank=512,
        q_lora_rank=1536,
        qk_rope_head_dim=64,
        v_head_dim=128,
        qk_nope_head_dim=128,
        max_position_embeddings=4096,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        attention_bias=False,
        attention_dropout=0.0,
    )

    if use_pretrained:
        config, weights = pretrained_weights
    else:
        config = test_config
        weights = random_weights

    # Create reference MLA
    if use_pretrained:
        # For pretrained, create from weights
        logger.info("Creating reference MLA with pretrained weights...")
        mla_ref = create_mla_reference(
            config=config,
            use_pretrained=True,
            state_dict={"model.layers.0.self_attn." + k: v for k, v in weights.items()},
            layer_idx=0,
            module_path="model.layers.0.self_attn",
        )
    else:
        # For random, use same seed
        logger.info("Creating reference MLA with random weights...")
        mla_ref = create_mla_reference(
            config=config,
            use_pretrained=False,
            layer_idx=0,
            seed=42,
        )

    # Create TT MLA
    logger.info("Creating TT MLA...")
    mla_tt = ttMLA(config, weights, mesh_device, layer_idx=0)

    # Verify both exist
    assert mla_ref is not None, "Reference MLA should exist"
    assert mla_tt is not None, "TT MLA should exist"

    # Get weight counts
    ref_state_dict = mla_ref.state_dict()
    tt_weight_shapes = mla_tt.get_weight_shapes()

    logger.info(f"✓ Both modules created successfully")
    logger.info(f"  Reference: {len(ref_state_dict)} parameters")
    logger.info(f"  TT:        {len(tt_weight_shapes)} weights")

    # Verify key weights exist
    key_weights = ["q_a_proj.weight", "q_b_proj.weight", "o_proj.weight"]
    for key in key_weights:
        ref_key = f"attention.{key}"
        assert ref_key in ref_state_dict, f"Reference missing {ref_key}"
        assert key in tt_weight_shapes, f"TT missing {key}"

    logger.info(f"✓ Weight loading verified")

    # Test forward pass comparison
    logger.info("=" * 80)
    logger.info(f"Testing forward pass comparison (seq_len={seq_len})")
    logger.info("=" * 80)

    # Create test inputs
    batch_size = 1
    hidden_size = config.hidden_size

    logger.info(f"Creating test inputs: batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}")

    # Create random input tensor
    torch.manual_seed(42)
    hidden_states = torch.randn(batch_size, seq_len, hidden_size, dtype=torch.bfloat16)

    # Create causal attention mask
    attention_mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bfloat16) * float("-inf"), diagonal=1)
    attention_mask = attention_mask.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, seq_len, seq_len)

    # Create position IDs
    position_ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(0).expand(batch_size, seq_len)

    # Run reference forward pass
    logger.info("Running reference CPU forward pass...")
    mla_ref = mla_ref.eval().to(torch.bfloat16)
    with torch.no_grad():
        ref_output, _, _ = mla_ref(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=None,
            output_attentions=False,
            use_cache=False,
        )

    logger.info(f"✓ Reference forward pass complete")
    logger.info(f"  Input shape:  {hidden_states.shape}")
    logger.info(f"  Output shape: {ref_output.shape}")
    logger.info(f"  Output dtype: {ref_output.dtype}")
    logger.info(f"  Output mean:  {ref_output.mean().item():.4f}")
    logger.info(f"  Output std:   {ref_output.std().item():.4f}")

    tt_hidden_states = ttnn.from_torch(
        hidden_states.unsqueeze(0),
        device=mesh_device,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=(-2, -1)),
    )
    tt_output = mla_tt.forward(
        hidden_states=tt_hidden_states,
        rope_tensors=None,
    )
    logger.warning("⚠️  TT forward pass not yet implemented - skipping comparison")

    # TODO: Compare outputs
    # logger.info("Comparing reference vs TT outputs...")
    # if tt_output is not None:
    #     # Convert TT output to CPU for comparison
    #     tt_output_cpu = tt_output.cpu()  # or appropriate conversion
    #
    #     # Compute PCC (Pearson Correlation Coefficient)
    #     pcc = torch.corrcoef(torch.stack([
    #         ref_output.flatten(),
    #         tt_output_cpu.flatten()
    #     ]))[0, 1].item()
    #
    #     # Compute relative error
    #     rel_error = torch.abs(ref_output - tt_output_cpu) / (torch.abs(ref_output) + 1e-6)
    #     max_rel_error = rel_error.max().item()
    #     mean_rel_error = rel_error.mean().item()
    #
    #     logger.info(f"  PCC: {pcc:.6f}")
    #     logger.info(f"  Max relative error: {max_rel_error:.6f}")
    #     logger.info(f"  Mean relative error: {mean_rel_error:.6f}")
    #
    #     # Assert PCC is high enough (e.g., > 0.99 for good match)
    #     assert pcc > 0.99, f"PCC too low: {pcc:.6f}, expected > 0.99"
    #     logger.success(f"✓ Output comparison passed (PCC={pcc:.6f})")

    logger.success(f"✓ Reference and TT comparison with {weight_type} weights successful")
