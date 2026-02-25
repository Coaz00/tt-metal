# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

import torch
from transformers.configuration_utils import PretrainedConfig

import ttnn


class MLASimple:
    def __init__(
        self,
        config: PretrainedConfig,
        state_dict: dict[str, torch.Tensor],
        mesh_device: ttnn.MeshDevice,
        layer_idx: int = 0,
    ):
        self.config = config
        self.mesh_device = mesh_device
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

        q_a_proj = state_dict["q_a_proj.weight"]
        q_a_proj = q_a_proj.transpose(-2, -1)
        tp_shard_dim = 0
        mesh_mapper = ttnn.ShardTensor2dMesh(
            self.mesh_device, mesh_shape=tuple(self.mesh_device.shape), dims=(None, tp_shard_dim)
        )
        self.q_a_proj_weight = self._to_tt_tensor(q_a_proj, ttnn.bfloat8_b, ttnn.TILE_LAYOUT, mesh_mapper)
        print(f"q_a_proj_weight per device shape: {self.q_a_proj_weight.shape}")

        self.q_a_layernorm_weight = self._to_tt_tensor(
            state_dict["q_a_layernorm.weight"],
            ttnn.bfloat16,
            ttnn.TILE_LAYOUT,
            ttnn.ReplicateTensorToMesh(self.mesh_device),
        )  # check this
        self.kv_a_layernorm_weight = self._to_tt_tensor(
            state_dict["kv_a_layernorm.weight"],
            ttnn.bfloat16,
            ttnn.TILE_LAYOUT,
            ttnn.ReplicateTensorToMesh(self.mesh_device),
        )  # check this
        self.q_b_proj_weight = self._to_tt_tensor(
            state_dict["q_b_proj.weight"],
            ttnn.bfloat8_b,
            ttnn.TILE_LAYOUT,
            ttnn.ReplicateTensorToMesh(self.mesh_device),
        )

        # KV projection weights
        self.kv_a_proj_with_mqa_weight = self._to_tt_tensor(
            state_dict["kv_a_proj_with_mqa.weight"],
            ttnn.bfloat8_b,
            ttnn.TILE_LAYOUT,
            ttnn.ReplicateTensorToMesh(self.mesh_device),
        )
        self.kv_b_proj_weight = self._to_tt_tensor(
            state_dict["kv_b_proj.weight"],
            ttnn.bfloat8_b,
            ttnn.TILE_LAYOUT,
            ttnn.ReplicateTensorToMesh(self.mesh_device),
        )

        # Output projection weight
        self.o_proj_weight = self._to_tt_tensor(
            state_dict["o_proj.weight"], ttnn.bfloat8_b, ttnn.TILE_LAYOUT, ttnn.ReplicateTensorToMesh(self.mesh_device)
        )

        print(f"✓ Loaded {len(state_dict)} weights to TT device")

    def _to_tt_tensor(
        self, tensor: torch.Tensor, dtype: ttnn.DataType, layout: ttnn.Layout, mesh_mapper: ttnn.TensorToMesh
    ) -> ttnn.Tensor:
        return ttnn.from_torch(
            tensor,
            device=self.mesh_device,
            dtype=dtype,
            layout=layout,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=mesh_mapper,
        )

    def get_weight_shapes(self) -> dict[str, tuple]:
        return {
            "q_a_proj.weight": tuple(self.q_a_proj_weight.shape),
            "q_a_layernorm.weight": tuple(self.q_a_layernorm_weight.shape),
            "q_b_proj.weight": tuple(self.q_b_proj_weight.shape),
            "kv_a_proj_with_mqa.weight": tuple(self.kv_a_proj_with_mqa_weight.shape),
            "kv_a_layernorm.weight": tuple(self.kv_a_layernorm_weight.shape),
            "kv_b_proj.weight": tuple(self.kv_b_proj_weight.shape),
            "o_proj.weight": tuple(self.o_proj_weight.shape),
        }
