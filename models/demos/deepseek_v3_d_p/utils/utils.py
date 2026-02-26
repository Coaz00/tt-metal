import torch
from transformers.configuration_utils import PretrainedConfig

import ttnn
from models.demos.deepseek_v3.tt.rope import RotarySetup


def get_rope_tensors(
    hf_config: PretrainedConfig,
    batch_size_per_row: int,
    seq_len: int,
    position_ids: torch.Tensor | None,
    mesh_device: ttnn.MeshDevice,
) -> dict[str, ttnn.Tensor]:
    rope_setup = RotarySetup(
        device=mesh_device,
        batch_size_per_row=batch_size_per_row,
        hf_config=hf_config,
    )
    if position_ids is None:
        return rope_setup.get_rot_mats_table(seq_len)
    return rope_setup.get_rot_mats(position_ids)
