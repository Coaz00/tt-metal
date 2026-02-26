# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""
Test file for MLA module with support for both CPU reference and device implementations.
Supports testing with both pretrained weights and random weights.
"""

import pytest
import torch
from loguru import logger

from models.demos.deepseek_v3.reference.configuration_deepseek import DeepseekV3Config
from models.demos.deepseek_v3.utils.config_helpers import sub_state_dict
from models.demos.deepseek_v3.utils.test_utils import dequantize_state_dict
from models.demos.deepseek_v3_d_p.reference.mla_reference import create_mla_reference


def create_attention_mask(batch_size: int, seq_len: int, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """
    Create causal attention mask for testing.

    Args:
        batch_size: Batch size
        seq_len: Sequence length
        dtype: Data type for mask

    Returns:
        Attention mask of shape [batch_size, 1, seq_len, seq_len]
    """
    # Create causal mask (lower triangular)
    mask = torch.triu(torch.ones(seq_len, seq_len, dtype=dtype) * float("-inf"), diagonal=1)
    return mask.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, seq_len, seq_len)


def create_position_ids(batch_size: int, seq_len: int) -> torch.Tensor:
    """
    Create position IDs for testing.

    Args:
        batch_size: Batch size
        seq_len: Sequence length

    Returns:
        Position IDs of shape [batch_size, seq_len]
    """
    return torch.arange(seq_len, dtype=torch.long).unsqueeze(0).expand(batch_size, seq_len)


class TestMLAReference:
    """Test suite for MLA reference implementation."""

    @pytest.fixture
    def test_config(self, hf_config):
        """Create a simplified config for testing."""
        config = DeepseekV3Config(
            vocab_size=hf_config.vocab_size,
            hidden_size=hf_config.hidden_size,
            num_attention_heads=hf_config.num_attention_heads,
            num_key_value_heads=hf_config.num_key_value_heads,
            kv_lora_rank=hf_config.kv_lora_rank,
            q_lora_rank=hf_config.q_lora_rank,
            qk_rope_head_dim=hf_config.qk_rope_head_dim,
            v_head_dim=hf_config.v_head_dim,
            qk_nope_head_dim=hf_config.qk_nope_head_dim,
            max_position_embeddings=hf_config.max_position_embeddings,
            rms_norm_eps=hf_config.rms_norm_eps,
            rope_theta=hf_config.rope_theta,
            attention_bias=hf_config.attention_bias,
            attention_dropout=hf_config.attention_dropout,
        )
        # Add quantization_config if it exists in hf_config (needed for dequantize_state_dict)
        if hasattr(hf_config, "quantization_config") and hf_config.quantization_config is not None:
            config.quantization_config = hf_config.quantization_config
        else:
            # Default quantization config if not present
            config.quantization_config = {"weight_block_size": [128, 128]}
        return config

    @pytest.mark.parametrize("use_pretrained", [True, False], ids=["pretrained_weights", "random_weights"])
    @pytest.mark.parametrize("batch_size", [1, 2])
    @pytest.mark.parametrize("seq_len", [32, 128])
    def test_mla_forward_cpu(
        self,
        test_config,
        use_pretrained,
        batch_size,
        seq_len,
        model_path,  # noqa: ARG002
        state_dict,
    ):
        """
        Test MLA forward pass on CPU with both pretrained and random weights.

        Args:
            test_config: Test configuration
            use_pretrained: Whether to use pretrained weights
            batch_size: Batch size for testing
            seq_len: Sequence length for testing
            model_path: Path to model directory (from fixture)
            state_dict: Loaded state dict (from fixture)
        """
        logger.info(
            f"Testing MLA CPU forward pass: use_pretrained={use_pretrained}, batch_size={batch_size}, seq_len={seq_len}"
        )

        # Create MLA module
        layer_idx = 0
        module_path = "model.layers.0.self_attn"

        if use_pretrained:
            # Load and dequantize state dict for the specific layer
            layer_state_dict = sub_state_dict(state_dict, module_path + ".")
            dequantized_state_dict = dequantize_state_dict(layer_state_dict, test_config)

            mla_module = create_mla_reference(
                config=test_config,
                use_pretrained=True,
                state_dict={module_path + "." + k: v for k, v in dequantized_state_dict.items()},
                layer_idx=layer_idx,
                module_path=module_path,
            )
        else:
            mla_module = create_mla_reference(
                config=test_config,
                use_pretrained=False,
                layer_idx=layer_idx,
                seed=42,
            )

        # Set to eval mode and bfloat16
        mla_module = mla_module.eval().to(torch.bfloat16)

        # Create inputs
        hidden_states = torch.randn(batch_size, seq_len, test_config.hidden_size, dtype=torch.bfloat16)
        attention_mask = create_attention_mask(batch_size, seq_len, dtype=torch.bfloat16)
        position_ids = create_position_ids(batch_size, seq_len)

        # Forward pass
        with torch.no_grad():
            output, _, _ = mla_module(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=None,
                output_attentions=False,
                use_cache=False,
            )

        # Check output shape
        expected_shape = (batch_size, seq_len, test_config.hidden_size)
        assert output.shape == expected_shape, f"Expected output shape {expected_shape}, got {output.shape}"

        # Check output is not NaN or Inf
        assert not torch.isnan(output).any(), "Output contains NaN values"
        assert not torch.isinf(output).any(), "Output contains Inf values"

        logger.info(f"✓ MLA CPU forward pass successful: output shape {output.shape}")

    @pytest.mark.parametrize("use_pretrained", [True, False], ids=["pretrained_weights", "random_weights"])
    def test_mla_decode_mode(
        self,
        test_config,
        use_pretrained,
        model_path,  # noqa: ARG002
        state_dict,
    ):
        """
        Test MLA in decode mode (sequence length = 1) with KV caching.

        Args:
            test_config: Test configuration
            use_pretrained: Whether to use pretrained weights
            model_path: Path to model directory (from fixture)
            state_dict: Loaded state dict (from fixture)
        """
        logger.info(f"Testing MLA decode mode: use_pretrained={use_pretrained}")

        # Create MLA module
        layer_idx = 0
        module_path = "model.layers.0.self_attn"

        if use_pretrained:
            layer_state_dict = sub_state_dict(state_dict, module_path + ".")
            dequantized_state_dict = dequantize_state_dict(layer_state_dict, test_config)

            mla_module = create_mla_reference(
                config=test_config,
                use_pretrained=True,
                state_dict={module_path + "." + k: v for k, v in dequantized_state_dict.items()},
                layer_idx=layer_idx,
                module_path=module_path,
            )
        else:
            mla_module = create_mla_reference(
                config=test_config,
                use_pretrained=False,
                layer_idx=layer_idx,
                seed=42,
            )

        mla_module = mla_module.eval().to(torch.bfloat16)

        # Simulate decode mode with seq_len=1
        batch_size = 1
        seq_len = 1
        hidden_states = torch.randn(batch_size, seq_len, test_config.hidden_size, dtype=torch.bfloat16)
        attention_mask = create_attention_mask(batch_size, seq_len, dtype=torch.bfloat16)
        position_ids = torch.tensor([[0]], dtype=torch.long)

        # Forward pass
        with torch.no_grad():
            output, _, _ = mla_module(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=None,
                output_attentions=False,
                use_cache=True,
            )

        # Check output
        assert output.shape == (batch_size, seq_len, test_config.hidden_size)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

        logger.info(f"✓ MLA decode mode successful: output shape {output.shape}")

    @pytest.mark.parametrize("use_pretrained", [False], ids=["random_weights"])
    def test_mla_reproducibility(
        self,
        test_config,
        use_pretrained,  # noqa: ARG002
    ):
        """
        Test that MLA with random weights produces reproducible results with same seed.

        Args:
            test_config: Test configuration
            use_pretrained: Whether to use pretrained weights (False for this test)
        """
        logger.info("Testing MLA reproducibility with random weights")

        seed = 12345
        batch_size = 1
        seq_len = 32

        # Create two modules with same seed
        mla_module1 = (
            create_mla_reference(
                config=test_config,
                use_pretrained=False,
                layer_idx=0,
                seed=seed,
            )
            .eval()
            .to(torch.bfloat16)
        )

        mla_module2 = (
            create_mla_reference(
                config=test_config,
                use_pretrained=False,
                layer_idx=0,
                seed=seed,
            )
            .eval()
            .to(torch.bfloat16)
        )

        # Create same input with fixed seed
        torch.manual_seed(seed)
        hidden_states = torch.randn(batch_size, seq_len, test_config.hidden_size, dtype=torch.bfloat16)
        attention_mask = create_attention_mask(batch_size, seq_len, dtype=torch.bfloat16)
        position_ids = create_position_ids(batch_size, seq_len)

        # Forward pass on both modules
        with torch.no_grad():
            output1, _, _ = mla_module1(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=None,
                output_attentions=False,
                use_cache=False,
            )

            output2, _, _ = mla_module2(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=None,
                output_attentions=False,
                use_cache=False,
            )

        # Check outputs are identical
        assert torch.allclose(output1, output2, rtol=1e-5, atol=1e-5), "Outputs are not reproducible with same seed"

        logger.info("✓ MLA reproducibility test passed")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
