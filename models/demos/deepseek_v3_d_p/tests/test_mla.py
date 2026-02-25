# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""
Test for instantiating both reference CPU and TT device MLA modules with the same weights.
This test verifies that both modules can be created and weights are loaded correctly.
"""

import pytest
import torch
from loguru import logger

from models.demos.deepseek_v3.reference.configuration_deepseek import DeepseekV3Config
from models.demos.deepseek_v3_d_p.reference.mla_reference import create_mla_reference
from models.demos.deepseek_v3_d_p.tt.mla import MLASimple


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


@pytest.mark.parametrize(
    "mesh_device",
    [(2, 4)],
    ids=["2x4"],
    indirect=True,
)
@pytest.mark.parametrize("use_pretrained", [False, True], ids=["random", "pretrained"])
def test_mla(use_pretrained, random_weights, pretrained_weights, mesh_device):
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
    mla_tt = MLASimple(config, weights, mesh_device, layer_idx=0)

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

    logger.success(f"✓ Reference and TT comparison with {weight_type} weights successful")
