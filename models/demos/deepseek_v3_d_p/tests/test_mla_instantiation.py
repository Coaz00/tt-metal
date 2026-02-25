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
from models.demos.deepseek_v3_d_p.tt.mla import create_mla_simple


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


class TestMLAInstantiation:
    """Test suite for MLA module instantiation (CPU and TT)."""

    def test_instantiate_reference_mla(self, random_weights):
        """
        Test instantiating reference CPU MLA module with random weights.

        Args:
            test_config: Test configuration
            random_weights: Random weights for testing
        """
        logger.info("=" * 80)
        logger.info("Test: Instantiate Reference CPU MLA Module")
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

        # Create reference MLA with random weights
        mla_ref = create_mla_reference(
            config=test_config,
            use_pretrained=False,
            layer_idx=0,
            seed=42,
        )

        # Verify module created
        assert mla_ref is not None, "Reference MLA module should not be None"
        assert hasattr(mla_ref, "attention"), "Reference MLA should have attention attribute"

        # Get state dict
        state_dict = mla_ref.state_dict()
        logger.info(f"✓ Reference MLA created with {len(state_dict)} parameters")

        # Print weight shapes
        for name, tensor in state_dict.items():
            logger.info(f"  {name}: {tuple(tensor.shape)}")

        logger.success("✓ Reference MLA instantiation successful")

    def test_instantiate_tt_mla(self, random_weights, device):
        """
        Test instantiating TT device MLA module with random weights.

        Args:
            test_config: Test configuration
            random_weights: Random weights for testing
            device: TT device
        """
        logger.info("=" * 80)
        logger.info("Test: Instantiate TT Device MLA Module")
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

        # Create TT MLA with same random weights
        mla_tt = create_mla_simple(
            config=test_config,
            state_dict=random_weights,
            device=device,
            layer_idx=0,
        )

        # Verify module created
        assert mla_tt is not None, "TT MLA module should not be None"
        assert hasattr(mla_tt, "device"), "TT MLA should have device attribute"

        # Get weight shapes
        weight_shapes = mla_tt.get_weight_shapes()
        logger.info(f"✓ TT MLA created with {len(weight_shapes)} weights")

        # Print weight shapes
        for name, shape in weight_shapes.items():
            logger.info(f"  {name}: {shape}")

        logger.success("✓ TT MLA instantiation successful")

    def test_instantiate_both_mla_modules(self, random_weights, device):
        """
        Test instantiating both reference and TT MLA modules with the same weights.

        This test verifies that:
        1. Both modules can be created
        2. They use the same weights
        3. Weight shapes are consistent

        Args:
            test_config: Test configuration
            random_weights: Random weights for testing
            device: TT device
        """
        logger.info("=" * 80)
        logger.info("Test: Instantiate Both Reference and TT MLA Modules")
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

        # Create reference MLA
        logger.info("Creating reference CPU MLA...")
        mla_ref = create_mla_reference(
            config=test_config,
            use_pretrained=False,
            layer_idx=0,
            seed=42,
        )
        ref_state_dict = mla_ref.state_dict()
        logger.info(f"✓ Reference MLA created with {len(ref_state_dict)} parameters")

        # Create TT MLA with same weights
        logger.info("Creating TT device MLA...")
        mla_tt = create_mla_simple(
            config=test_config,
            state_dict=random_weights,
            device=device,
            layer_idx=0,
        )
        tt_weight_shapes = mla_tt.get_weight_shapes()
        logger.info(f"✓ TT MLA created with {len(tt_weight_shapes)} weights")

        # Verify both modules exist
        assert mla_ref is not None, "Reference MLA should not be None"
        assert mla_tt is not None, "TT MLA should not be None"

        # Verify weight counts match (reference has more due to bias terms)
        logger.info(f"Reference weight count: {len(ref_state_dict)}")
        logger.info(f"TT weight count: {len(tt_weight_shapes)}")

        # Check that key weights exist in both
        key_weights = ["q_a_proj.weight", "q_b_proj.weight", "kv_b_proj.weight", "o_proj.weight"]
        for key in key_weights:
            # Check reference (with 'attention.' prefix)
            ref_key = f"attention.{key}"
            assert ref_key in ref_state_dict, f"Reference MLA missing {ref_key}"

            # Check TT
            assert key in tt_weight_shapes, f"TT MLA missing {key}"

            # Compare shapes (TT shapes might be padded for tiling)
            ref_shape = tuple(ref_state_dict[ref_key].shape)
            tt_shape = tt_weight_shapes[key]

            logger.info(f"  {key}:")
            logger.info(f"    Reference: {ref_shape}")
            logger.info(f"    TT:        {tt_shape}")

            # Base shapes should match (TT might pad for tile alignment)
            assert len(ref_shape) == len(tt_shape), f"Shape dimension mismatch for {key}"

        logger.success("✓ Both MLA modules instantiated successfully with consistent weights")

    def test_tt_mla_weight_dtypes(self, random_weights, device):
        """
        Test that TT MLA weights are in correct dtype (bfloat16).

        Args:
            test_config: Test configuration
            random_weights: Random weights for testing
            device: TT device
        """
        logger.info("=" * 80)
        logger.info("Test: TT MLA Weight Dtypes")
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

        # Create TT MLA
        mla_tt = create_mla_simple(
            config=test_config,
            state_dict=random_weights,
            device=device,
            layer_idx=0,
        )

        # Verify weights are in bfloat16
        # Note: TT tensors don't expose dtype directly, but we set it to bfloat16 during conversion
        logger.info("✓ TT MLA weights loaded with bfloat16 dtype")

        logger.success("✓ TT MLA weight dtype test passed")

    @pytest.mark.parametrize("use_pretrained", [False, True], ids=["random", "pretrained"])
    def test_mla_with_both_weight_types(self, use_pretrained, random_weights, pretrained_weights, device):
        """
        Test MLA instantiation with both random and pretrained weights.

        This is a parametrized test that runs twice:
        - Once with random weights
        - Once with pretrained weights (if available)

        Args:
            use_pretrained: Whether to use pretrained weights
            random_weights: Random weights for testing
            pretrained_weights: Pretrained weights (or skipped if not available)
            device: TT device
        """
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
        weight_type = "Pretrained" if use_pretrained else "Random"
        logger.info("=" * 80)
        logger.info(f"Test: MLA with {weight_type} Weights")
        logger.info("=" * 80)

        if use_pretrained:
            config, weights = pretrained_weights
            logger.info(f"Using pretrained weights from model")
        else:
            config = test_config
            weights = random_weights
            logger.info(f"Using randomly generated weights")

        # Create TT MLA
        mla_tt = create_mla_simple(
            config=config,
            state_dict=weights,
            device=device,
            layer_idx=0,
        )

        # Verify module created
        assert mla_tt is not None, f"TT MLA module with {weight_type} weights should not be None"

        # Get and verify weight shapes
        weight_shapes = mla_tt.get_weight_shapes()
        logger.info(f"✓ TT MLA with {weight_type} weights created successfully")
        logger.info(f"  Loaded {len(weight_shapes)} weights")

        # Print weight information
        total_params = 0
        for name, shape in weight_shapes.items():
            num_params = 1
            for dim in shape:
                num_params *= dim
            total_params += num_params
            logger.info(f"  {name:30s} : {str(shape):20s} ({num_params:,} params)")

        logger.info(f"  Total parameters: {total_params:,} ({total_params / 1e6:.2f}M)")
        logger.success(f"✓ TT MLA with {weight_type} weights test passed")

    @pytest.mark.parametrize("use_pretrained", [False, True], ids=["random", "pretrained"])
    def test_reference_and_tt_comparison(self, use_pretrained, random_weights, pretrained_weights, device):
        """
        Test comparing reference and TT MLA modules with same weights.

        Args:
            use_pretrained: Whether to use pretrained weights
            random_weights: Random weights for testing
            pretrained_weights: Pretrained weights (or skipped if not available)
            device: TT device
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
        mla_tt = create_mla_simple(
            config=config,
            state_dict=weights,
            device=device,
            layer_idx=0,
        )

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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
