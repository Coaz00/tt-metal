# TT MLA Module for DeepSeek V3

Simple single-device TT implementation of Multi-Latent Attention (MLA) for DeepSeek V3.

## Overview

This module provides a simplified TT (Tenstorrent) implementation of the MLA attention mechanism that:
- ✅ Loads weights from PyTorch state dict
- ✅ Converts to TT tensors using `ttnn.from_torch()`
- ✅ Runs on a single device (no mesh)
- ✅ No complex caching or configurations
- ✅ Designed for testing and development

## Files

- **`mla.py`**: Simple single-device TT MLA module
- **`../tests/test_mla_instantiation.py`**: Tests for instantiating reference and TT modules

## Usage

### Creating TT MLA Module

```python
import ttnn
from models.demos.deepseek_v3.reference.configuration_deepseek import DeepseekV3Config
from models.demos.deepseek_v3_d_p.tt.mla import create_mla_simple

# Setup
device = ttnn.open_device(device_id=0)
config = DeepseekV3Config()

# Create state dict with weights (dequantized, bfloat16)
state_dict = {
    "q_a_proj.weight": ...,
    "q_a_layernorm.weight": ...,
    "q_b_proj.weight": ...,
    "kv_a_proj_with_mqa.weight": ...,
    "kv_a_layernorm.weight": ...,
    "kv_b_proj.weight": ...,
    "o_proj.weight": ...,
}

# Create MLA module
mla_tt = create_mla_simple(
    config=config,
    state_dict=state_dict,
    device=device,
    layer_idx=0,
)

# Get weight shapes
shapes = mla_tt.get_weight_shapes()
print(shapes)

# Cleanup
ttnn.close_device(device)
```

## Running Tests

### Test MLA Instantiation

```bash
cd /localdev/ppopovic/tt-metal

# Run all instantiation tests (with random weights)
pytest models/demos/deepseek_v3_d_p/tests/test_mla_instantiation.py -v

# Run specific test
pytest models/demos/deepseek_v3_d_p/tests/test_mla_instantiation.py::TestMLAInstantiation::test_instantiate_both_mla_modules -v

# Run only random weight tests
pytest models/demos/deepseek_v3_d_p/tests/test_mla_instantiation.py -v -k "random"

# Run only pretrained weight tests (requires model)
pytest models/demos/deepseek_v3_d_p/tests/test_mla_instantiation.py -v -k "pretrained"
```

### Available Tests

#### Basic Tests (Random Weights)
1. **`test_instantiate_reference_mla`**: Creates reference CPU MLA module
2. **`test_instantiate_tt_mla`**: Creates TT device MLA module
3. **`test_instantiate_both_mla_modules`**: Creates both and verifies consistency
4. **`test_tt_mla_weight_dtypes`**: Verifies weight dtypes

#### Parametrized Tests (Random + Pretrained)
5. **`test_mla_with_both_weight_types`**: Tests with both random and pretrained weights
6. **`test_reference_and_tt_comparison`**: Compares reference and TT with same weights

**Note**: Pretrained weight tests **automatically download weights** from HuggingFace if not found locally. First run will download ~3-5GB to `~/.cache/huggingface/`. Subsequent runs use cached weights instantly.

### Pretrained Weights (Automatic)

Tests automatically download pretrained weights from HuggingFace when needed. No manual setup required!

```bash
# Just run tests - weights download automatically on first run
pytest models/demos/deepseek_v3_d_p/tests/test_mla_instantiation.py -v

# First run: downloads ~3-5GB (layer 0 only)
# Subsequent runs: instant (uses cached weights)
```

#### Optional: Pre-download or Use Custom Weights

If you prefer to pre-download or use existing weights:

```bash
# Option 1: Set environment variable to existing weights
export DEEPSEEK_V3_HF_MODEL=/path/to/your/deepseek-r1-0528

# Option 2: Pre-download using standalone script
python models/demos/deepseek_v3_d_p/reference/test_mla_standalone.py --mode pretrained

# Option 3: Manual download with huggingface-cli
huggingface-cli download deepseek-ai/DeepSeek-R1-0528 \
    --include "*.json" "*.py" "*-00001-of-*.safetensors" "*-00002-of-*.safetensors" "*-00003-of-*.safetensors" \
    --local-dir models/demos/deepseek_v3/reference \
    --local-dir-use-symlinks False
```

## Weight Format

The TT MLA module expects a state dict with these keys:

| Weight Name | Shape | Description |
|-------------|-------|-------------|
| `q_a_proj.weight` | (q_lora_rank, hidden_size) | Query compression |
| `q_a_layernorm.weight` | (q_lora_rank,) | Query norm |
| `q_b_proj.weight` | (num_heads × q_head_dim, q_lora_rank) | Query expansion |
| `kv_a_proj_with_mqa.weight` | (kv_lora_rank + rope_dim, hidden_size) | KV compression |
| `kv_a_layernorm.weight` | (kv_lora_rank,) | KV norm |
| `kv_b_proj.weight` | (num_heads × (nope_dim + v_dim), kv_lora_rank) | KV expansion |
| `o_proj.weight` | (hidden_size, num_heads × v_head_dim) | Output projection |

**Requirements:**
- All weights must be in `torch.bfloat16` dtype
- Weights should be dequantized (not FP8)
- Shapes must match the config dimensions

## Example: Load from DeepSeek-R1-0528

```python
from pathlib import Path
from models.demos.deepseek_v3.utils.test_utils import load_state_dict, dequantize_state_dict
from models.demos.deepseek_v3.utils.config_helpers import sub_state_dict
from transformers import AutoConfig

# Load config
model_path = Path("/path/to/deepseek-r1-0528")
config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)

# Load and extract layer 0 attention weights
state_dict = load_state_dict(model_path, "")
layer_state_dict = sub_state_dict(state_dict, "model.layers.0.self_attn.")
dequantized_weights = dequantize_state_dict(layer_state_dict, config)

# Create TT MLA
device = ttnn.open_device(device_id=0)
mla_tt = create_mla_simple(
    config=config,
    state_dict=dequantized_weights,
    device=device,
    layer_idx=0,
)
```

## Architecture

### MLA (Multi-Latent Attention) Structure

```
Input (hidden_size=7168)
    │
    ├─> Q Path:
    │   ├─> q_a_proj: 7168 → 1536 (compression)
    │   ├─> q_a_layernorm: RMSNorm(1536)
    │   └─> q_b_proj: 1536 → 24576 (128 heads × 192 dim)
    │
    ├─> KV Path:
    │   ├─> kv_a_proj_with_mqa: 7168 → 576 (512 + 64 rope)
    │   ├─> kv_a_layernorm: RMSNorm(512)
    │   └─> kv_b_proj: 512 → 32768 (K + V expansion)
    │
    └─> Output:
        └─> o_proj: 16384 → 7168 (back to hidden size)
```

**Total Parameters**: ~187M (356.88 MB in bfloat16)

## Limitations

This is a **simplified implementation** for development and testing:
- ❌ No forward pass implemented yet
- ❌ No multi-device / mesh support
- ❌ No KV caching
- ❌ No optimized kernels
- ❌ No async operations

For production inference, use the full `MLA1D` or `MLA2D` implementations in `models/demos/deepseek_v3/tt/mla/`.

## Next Steps

1. **Add Forward Pass**: Implement the actual MLA computation
2. **Compare with Reference**: Verify TT outputs match CPU reference
3. **Measure PCC**: Calculate Pearson Correlation Coefficient
4. **Optimize**: Add performance optimizations

## Related Files

- Reference implementation: `models/demos/deepseek_v3_d_p/reference/mla_reference.py`
- Full TT implementation: `models/demos/deepseek_v3/tt/mla/mla1d.py`
- Tests: `models/demos/deepseek_v3_d_p/tests/test_mla_instantiation.py`
