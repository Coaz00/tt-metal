# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

import os
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.deepseek_v3.tt.mla.mla2d import MLA2D
from models.demos.deepseek_v3.tt.mtp import MTP2D
from models.demos.deepseek_v3.utils.config_helpers import USERS_PER_ROW, sub_state_dict
from models.demos.deepseek_v3.utils.run_config import create_run_config
from models.demos.deepseek_v3.utils.test_utils import (
    assert_hidden_dim_pcc,
    get_model_config,
    get_rope_tensors,
    get_test_weight_config,
)

MTP_PCC_REQUIRED = float(os.getenv("DEEPSEEK_MTP_PCC_REQUIRED", "0.999"))
MTP_ENABLE_PCC_CHECK = os.getenv("DEEPSEEK_MTP_RUN_PCC", "1") == "1"
MTP_ENABLE_E2E_PERF_CHECK = os.getenv("DEEPSEEK_MTP_RUN_E2E_PERF", "0") == "1"
MTP_E2E_MIN_TOKENS_PER_SEC = float(os.getenv("DEEPSEEK_MTP_E2E_MIN_TOKENS_PER_SEC", "0"))
MTP_TRACE_REGION_SIZE = int(os.getenv("DEEPSEEK_MTP_TRACE_REGION_SIZE", "134217728"))


def _get_mtp_state_dict(state_dict: dict[str, torch.Tensor], hf_config: Any) -> dict[str, torch.Tensor]:
    mtp_layer_idx = int(hf_config.num_hidden_layers)
    mtp_state_dict = sub_state_dict(state_dict, f"model.layers.{mtp_layer_idx}.")
    if len(mtp_state_dict) == 0:
        pytest.skip(f"No MTP layer keys found under model.layers.{mtp_layer_idx}.")
    return mtp_state_dict


def _run_mtp_forward(
    mode: str,
    enable_trace: bool,
    mtp_hf_config: Any,
    mtp_state_dict: dict[str, torch.Tensor],
    torch_hidden: torch.Tensor,
    torch_token_ids: torch.Tensor,
    position_ids: torch.Tensor,
    user_id: int,
    cache_path: Path,
    mesh_device: ttnn.MeshDevice,
    ccl: Any,
    force_recalculate_weight_config: bool,
) -> tuple[torch.Tensor, float, int]:
    paged_config = MLA2D.get_valid_paged_config(mtp_hf_config.max_seq_len, USERS_PER_ROW, mesh_device.shape[1])
    weight_config = get_test_weight_config(
        MTP2D,
        mtp_hf_config,
        (mtp_state_dict,),
        cache_path,
        mesh_device,
        force_recalculate_weight_config,
    )
    model_config = get_model_config(MTP2D, mode, mtp_hf_config, mesh_device)
    model_state = MTP2D.create_state(mtp_hf_config, paged_config, mesh_device, ccl)
    model_shared_state = MTP2D.create_shared_state(mtp_hf_config, mesh_device)
    run_config = create_run_config(model_config, weight_config, model_state, model_shared_state)

    tt_hidden = ttnn.from_torch(
        torch_hidden,
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(-2, -1), mesh_shape=tuple(mesh_device.shape)),
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        layout=ttnn.TILE_LAYOUT,
    )
    tt_token_ids = ttnn.from_torch(
        torch_token_ids,
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        dtype=ttnn.uint32,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )

    tt_position_ids = None
    if mode == "decode":
        tt_position_ids = ttnn.from_torch(
            position_ids.to(torch.int32),
            device=mesh_device,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0),
            dtype=ttnn.int32,
        )

    if mode == "decode":
        batch_size = int(torch_hidden.shape[2])
        num_devices = int(mesh_device.shape[0] * mesh_device.shape[1])
        assert batch_size % num_devices == 0, f"Decode batch size {batch_size} must divide by num_devices {num_devices}"
        batches_per_device = batch_size // num_devices
        assert batches_per_device > 0, "batches_per_device must be > 0 in decode mode"
        assert (
            paged_config.max_num_blocks % batches_per_device == 0
        ), f"max_num_blocks {paged_config.max_num_blocks} must divide by batches_per_device {batches_per_device}"
        blocks_per_batch = paged_config.max_num_blocks // batches_per_device
        page_table_host = torch.arange(batches_per_device * blocks_per_batch, dtype=torch.int32).reshape(
            batches_per_device, blocks_per_batch
        )
    else:
        page_table_host = torch.arange(paged_config.max_num_blocks, dtype=torch.int32).reshape(
            1, paged_config.max_num_blocks
        )
    tt_page_table = MLA2D.create_page_table(
        page_table=page_table_host, paged_config=paged_config, mesh_device=mesh_device
    )

    batch_size_per_row = torch_hidden.shape[2] // mesh_device.shape[0] if mode == "decode" else 1
    rope_tensors = get_rope_tensors(
        hf_config=mtp_hf_config,
        batch_size_per_row=batch_size_per_row,
        seq_len=torch_hidden.shape[2],
        position_ids=position_ids if mode == "decode" else None,
        mesh_device=mesh_device,
    )

    def run_op() -> ttnn.Tensor:
        if mode == "decode":
            assert tt_position_ids is not None
            return MTP2D.forward_decode(
                hidden_states=tt_hidden,
                token_ids=tt_token_ids,
                position_idxs=tt_position_ids,
                cfg=run_config,
                rope_tensors=rope_tensors,
                page_table=tt_page_table,
            )
        return MTP2D.forward_prefill(
            hidden_states=tt_hidden,
            token_ids=tt_token_ids,
            user_id=user_id,
            cfg=run_config,
            rope_tensors=rope_tensors,
            page_table=tt_page_table,
        )

    trace_id = None
    tt_output = None
    elapsed_s = 0.0
    try:
        if enable_trace:
            # Compile before trace capture.
            warmup_out = run_op()
            ttnn.deallocate(warmup_out)
            ttnn.synchronize_device(mesh_device)

            trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
            tt_output = run_op()
            ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
            ttnn.synchronize_device(mesh_device)

            start = time.perf_counter()
            ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=True)
            ttnn.synchronize_device(mesh_device)
            elapsed_s = time.perf_counter() - start
        else:
            # Compile/warmup once, then measure e2e latency of the second run.
            warmup_out = run_op()
            ttnn.deallocate(warmup_out)
            ttnn.synchronize_device(mesh_device)

            start = time.perf_counter()
            tt_output = run_op()
            ttnn.synchronize_device(mesh_device)
            elapsed_s = time.perf_counter() - start

        tt_output_torch = ttnn.to_torch(
            tt_output,
            mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(-2, -1), mesh_shape=tuple(mesh_device.shape)),
        )
    finally:
        if tt_output is not None:
            ttnn.deallocate(tt_output)
        if trace_id is not None:
            ttnn.release_trace(mesh_device, trace_id)
        if tt_position_ids is not None:
            ttnn.deallocate(tt_position_ids)
        ttnn.deallocate(tt_hidden)
        ttnn.deallocate(tt_token_ids)
        ttnn.deallocate(tt_page_table)

    tokens_processed = int(torch_hidden.shape[2])
    return tt_output_torch, elapsed_s, tokens_processed


@pytest.mark.timeout(1800)
@pytest.mark.parametrize(
    "device_params",
    [
        {"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": MTP_TRACE_REGION_SIZE},
    ],
    indirect=True,
)
@pytest.mark.parametrize("mode, seq_len, batch_size_per_row", [("decode", 1, USERS_PER_ROW), ("prefill", 128, 1)])
@pytest.mark.parametrize("enable_trace", [False, True])
@pytest.mark.requires_device(["TG", "DUAL", "QUAD"])
def test_mtp_forward_pass(
    mode: str,
    seq_len: int,
    batch_size_per_row: int,
    enable_trace: bool,
    hf_config: Any,
    hf_config_short: Any,
    cache_path: Path,
    mesh_device: ttnn.MeshDevice,
    ccl: Any,
    force_recalculate_weight_config: bool,
    set_deterministic_env: Any,
    state_dict: dict[str, torch.Tensor],
):
    mtp_hf_config = deepcopy(hf_config)
    mtp_hf_config.max_seq_len = hf_config_short.max_seq_len

    mtp_state_dict = _get_mtp_state_dict(state_dict, hf_config)

    batch_size = batch_size_per_row * mesh_device.shape[0] if mode == "decode" else 1
    token_count = seq_len if mode == "prefill" else batch_size

    torch_hidden = torch.randn(1, 1, token_count, mtp_hf_config.hidden_size).to(torch.bfloat16)
    torch_token_ids = torch.randint(
        low=0,
        high=mtp_hf_config.vocab_size - 1,
        size=(1, 1, token_count),
        dtype=torch.int32,
    )
    position_ids = (
        torch.randint(0, mtp_hf_config.max_seq_len - 1, (batch_size,), dtype=torch.int64)
        if mode == "decode"
        else torch.tensor([], dtype=torch.int64)
    )
    user_id = 0

    tt_output_torch, elapsed_s, tokens_processed = _run_mtp_forward(
        mode=mode,
        enable_trace=enable_trace,
        mtp_hf_config=mtp_hf_config,
        mtp_state_dict=mtp_state_dict,
        torch_hidden=torch_hidden,
        torch_token_ids=torch_token_ids,
        position_ids=position_ids,
        user_id=user_id,
        cache_path=cache_path,
        mesh_device=mesh_device,
        ccl=ccl,
        force_recalculate_weight_config=force_recalculate_weight_config,
    )

    if MTP_ENABLE_PCC_CHECK:
        if enable_trace:
            reference_output_torch, _, _ = _run_mtp_forward(
                mode=mode,
                enable_trace=False,
                mtp_hf_config=mtp_hf_config,
                mtp_state_dict=mtp_state_dict,
                torch_hidden=torch_hidden,
                torch_token_ids=torch_token_ids,
                position_ids=position_ids,
                user_id=user_id,
                cache_path=cache_path,
                mesh_device=mesh_device,
                ccl=ccl,
                force_recalculate_weight_config=force_recalculate_weight_config,
            )
        else:
            reference_output_torch = tt_output_torch

        assert_hidden_dim_pcc(tt_output_torch, reference_output_torch, pcc_required=MTP_PCC_REQUIRED)

    e2e_toks_per_s = tokens_processed / max(elapsed_s, 1e-6)
    logger.info(
        f"MTP mode={mode} trace={enable_trace} e2e_toks_per_s={e2e_toks_per_s:.3f} "
        f"tokens_processed={tokens_processed} elapsed_s={elapsed_s:.6f}"
    )

    if MTP_ENABLE_E2E_PERF_CHECK:
        assert (
            MTP_E2E_MIN_TOKENS_PER_SEC > 0
        ), "DEEPSEEK_MTP_E2E_MIN_TOKENS_PER_SEC must be > 0 when DEEPSEEK_MTP_RUN_E2E_PERF=1"
        assert (
            e2e_toks_per_s >= MTP_E2E_MIN_TOKENS_PER_SEC
        ), f"MTP e2e perf too low: {e2e_toks_per_s:.3f} < {MTP_E2E_MIN_TOKENS_PER_SEC:.3f}"


if __name__ == "__main__":
    pytest.main([__file__])
