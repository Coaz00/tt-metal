# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""
Simple single-device TT MLA module for DeepSeek V3.
This module loads weights and converts them to TT tensors without complex configurations.
"""

import torch
from transformers.configuration_utils import PretrainedConfig

import ttnn


class MLASimple:
    """
    Simple Multi-Latent Attention module for single device.

    This module:
    - Loads weights from a state dict
    - Converts them to TT tensors using ttnn.from_torch()
    - Runs on a single device (no mesh)
    - No caching or complex configurations

    Args:
        config: DeepSeek V3 configuration
        state_dict: Dictionary of weights (dequantized, in bfloat16)
        device: TT device to load weights onto
        layer_idx: Layer index (default: 0)
    """

    def __init__(
        self,
        config: PretrainedConfig,
        state_dict: dict[str, torch.Tensor],
        device: ttnn.Device,
        layer_idx: int = 0,
    ):
        self.config = config
        self.device = device
        self.layer_idx = layer_idx

        # Extract dimensions from config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.kv_lora_rank = config.kv_lora_rank
        self.q_lora_rank = config.q_lora_rank
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.v_head_dim = config.v_head_dim
        self.q_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim

        # Load weights to TT device
        self._load_weights(state_dict)

    def _load_weights(self, state_dict: dict[str, torch.Tensor]):
        """
        Load weights from state dict and convert to TT tensors.

        Expected keys in state_dict:
        - q_a_proj.weight
        - q_a_layernorm.weight
        - q_b_proj.weight
        - kv_a_proj_with_mqa.weight
        - kv_a_layernorm.weight
        - kv_b_proj.weight
        - o_proj.weight
        """
        # Query projection weights
        self.q_a_proj_weight = self._to_tt_tensor(state_dict["q_a_proj.weight"])
        self.q_a_layernorm_weight = self._to_tt_tensor(state_dict["q_a_layernorm.weight"])
        self.q_b_proj_weight = self._to_tt_tensor(state_dict["q_b_proj.weight"])

        # KV projection weights
        self.kv_a_proj_with_mqa_weight = self._to_tt_tensor(state_dict["kv_a_proj_with_mqa.weight"])
        self.kv_a_layernorm_weight = self._to_tt_tensor(state_dict["kv_a_layernorm.weight"])
        self.kv_b_proj_weight = self._to_tt_tensor(state_dict["kv_b_proj.weight"])

        # Output projection weight
        self.o_proj_weight = self._to_tt_tensor(state_dict["o_proj.weight"])

        print(f"✓ Loaded {len(state_dict)} weights to TT device")

    def _to_tt_tensor(self, tensor: torch.Tensor) -> ttnn.Tensor:
        """
        Convert a PyTorch tensor to a TT tensor on device.

        Args:
            tensor: PyTorch tensor to convert

        Returns:
            TT tensor on device
        """
        # Simple conversion without mesh mappers
        return ttnn.from_torch(
            tensor,
            device=self.device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def get_weight_shapes(self) -> dict[str, tuple]:
        """
        Get shapes of all loaded weights.

        Returns:
            Dictionary mapping weight names to shapes
        """
        return {
            "q_a_proj.weight": tuple(self.q_a_proj_weight.shape),
            "q_a_layernorm.weight": tuple(self.q_a_layernorm_weight.shape),
            "q_b_proj.weight": tuple(self.q_b_proj_weight.shape),
            "kv_a_proj_with_mqa.weight": tuple(self.kv_a_proj_with_mqa_weight.shape),
            "kv_a_layernorm.weight": tuple(self.kv_a_layernorm_weight.shape),
            "kv_b_proj.weight": tuple(self.kv_b_proj_weight.shape),
            "o_proj.weight": tuple(self.o_proj_weight.shape),
        }

    def __del__(self):
        """Cleanup when module is deleted."""
        # TT tensors will be automatically cleaned up


def create_mla_simple(
    config: PretrainedConfig,
    state_dict: dict[str, torch.Tensor],
    device: ttnn.Device,
    layer_idx: int = 0,
) -> MLASimple:
    """
    Factory function to create a simple MLA module.

    Args:
        config: DeepSeek V3 configuration
        state_dict: Dictionary of weights (dequantized, in bfloat16)
        device: TT device to load weights onto
        layer_idx: Layer index (default: 0)

    Returns:
        MLASimple instance with weights loaded

    Example:
        >>> device = ttnn.open_device(device_id=0)
        >>> mla = create_mla_simple(config, state_dict, device)
        >>> shapes = mla.get_weight_shapes()
    """
    return MLASimple(config, state_dict, device, layer_idx)
