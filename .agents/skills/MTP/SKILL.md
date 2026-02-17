---
name: mtp-bringup
description: Bring up multi-token prediction (speculative decoding) for TTNN transformer demos like DeepSeek R1, including second-token prediction, cache priming, accept/reject logic, and verification against non-MTP baseline outputs and acceptance-rate thresholds.
---

# MTP Bringup (TTNN DeepSeek)

## Define MTP Behavior
- Predict the next-next token using a lightweight predictor that consumes:
  - Base model hidden state at position t (pre-norm, before RMSNorm)
  - Predicted next token at position t+1
- Verify the prediction by running the base model once on a batched set:
  - Predicted next token (normal decode)
  - MTP-predicted token (speculative decode)

## Integrate Required Signals
- Capture pre-norm hidden state (`hidden_for_mtp`) in both prefill and decode.
- Avoid post-norm hidden states for MTP.
- Use a dedicated MTP module (embedding, norms, decoder block, head).
- Maintain correct concat order in MTP projection. For DeepSeek-R1: `[token_norm, hidden_norm]`.

## Prime MTP Prefill Cache
- Initialize MTP KV cache after base prefill.
- Align inputs to the MTP rule (hidden at t with token at t+1):
  - `hidden_shifted = hidden[:, :-1]`
  - `token_shifted = tokens[:, 1:]`
  - Trim RoPE `cos/sin` to `seq_len-1` (do not shift; just slice)
- Avoid padding the tail token/hidden. Padding biases acceptance and can hide alignment bugs.
- Build shifted tokens on host if TTNN concat/shard causes CCL reduce_scatter failures.
- For multi-device reduce_scatter, the MTP prefill sequence length must be divisible by the ring size. Use the padded `full_seq_len` from `_pad_batch` for MTP prefill, then trim after all_gather.

## Align Decode Positions
- Align MTP decode positions to the token being predicted.
- Use `positions_before` for the MTP candidate token, and `positions_before+1` for verifying the next-next token.

## Implement Accept/Reject Logic
- Accept when MTP next-next token equals base verified token.
- Rebatch verification in this order:
  - Batch 0: predicted next token (base)
  - Batch 1: MTP-predicted next-next token
- Advance positions by +2 on accept, +1 on reject.

## Verify End to End
- Run full-model baseline (non-MTP, greedy) and capture output JSON.
- Run full-model MTP (greedy) and require exact output match with baseline.
- Track accept rate (target ~0.8; investigate if <0.5).
- Do not enable MTP for teacher-forcing accuracy; compare baseline vs MTP outputs instead.
- MTP cannot be forced when using `--override-num-layers` (no MTP layer in truncated configs). Use full model for MTP verification.
- Always verify baseline outputs match the known-good reference commit (e.g., `f250fa...`) before judging MTP.

## Watch For Common Failures
- `TT_FATAL reduce_scatter ring_size` in MTP prefill:
  - Avoid mis-sharded inputs.
  - Build shifted tokens on host with identical mesh replication.
  - Ensure the MTP prefill seq_len is padded to the mesh ring size (see above).
- `TT_FATAL ND sharding requires number of chunks`:
  - Avoid creating MTP dummy tensors with a different sharding scheme.
  - Prefer deriving dummy tensors from slices of `hidden_tt` (same sharding) and zeroing via `ttnn.mul(..., 0.0)`.
- Low acceptance:
  - Fix decode position alignment.
  - Fix concat ordering.
  - Ensure pre-norm hidden is used.
  - Ensure MTP prefill uses hidden[t] with token[t+1] and trimmed RoPE.

## Useful Flags and Env
- `--mtp on|off|auto`
- `--compare-output <baseline_json>`
- `--min-mtp-accept-rate <float>`
- `DEEPSEEK_MTP_DEBUG=1` for detailed logs
- Set `DEEPSEEK_V3_MAX_SEQ_LEN=128` for quick testing runs.
- `DEEPSEEK_V3_MAX_SEQ_LEN=<N>` to clamp sequence length

## Likely Files to Touch (DeepSeek v3 Demo)
- `models/demos/deepseek_v3/tt/generator.py`
- `models/demos/deepseek_v3/tt/mtp.py`
- `models/demos/deepseek_v3/tt/model/row_batched_model.py`
