---
name: tt-machines
description: Operate Tenstorrent machines for TTNN workloads, including environment setup, reset/recovery procedures, and safe run workflows for demos and tests.
---

# Tenstorrent Machine Ops

## Set Up Environment
- Set `TT_METAL_HOME` to the repo root (example: `/home/jrock/wa/tt-metal`).
- Source metal setup and activate venv when required:
  - `source ../setup_metla.sh`
  - `source python_env/bin/activate`

## Reset When Needed
- Reset on hangs, TLB allocation errors, or device stuck states:
  - `/home/shared/scripts/reset.sh`
- Ensure `TT_METAL_HOME` is set before resetting.
- After `reset` completes successfully, wait 30 seconds before launching the next command.

## Recover From Common Errors
- `TT_FATAL` or CCL ring errors:
  - Reset and rerun.
  - Verify sequence length/sharding compatibility.
- Device allocation failures:
  - Reset.
  - Avoid concurrent large jobs on the same mesh.
- Missing environment:
  - Export `TT_METAL_HOME` before running `ds-run` or reset scripts.

## Run Workflow
1. Source setup and activate venv.
2. Run a small test (short layers or small max tokens).
3. Run the full demo only after quick validation passes.
4. Reset and retry if a hang or fatal error occurs.

## Logging and Timeouts
- Always capture stdout/stderr to log files when running demos.
- Use reasonable timeouts (10-30 min) for long runs and reset on hangs.

## Reference Commands
- Demo run: `/home/shared/scripts/ds-run python ...`
- Accuracy check: `/home/shared/scripts/ds-run pytest ...`
- Reset: `/home/shared/scripts/reset.sh`
