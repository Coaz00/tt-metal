#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""
Standalone script to test MLA reference implementation.
Can be run without pytest for quick testing and debugging.

Usage:
    # Test with random weights (no downloads)
    python test_mla_standalone.py --mode random

    # Test with pretrained weights (auto-downloads DeepSeek-R1-0528 from HuggingFace)
    python test_mla_standalone.py --mode pretrained

    # Test with local pretrained weights
    python test_mla_standalone.py --mode pretrained --model-path /path/to/deepseek-r1

    # Test with custom batch size and sequence length
    python test_mla_standalone.py --mode random --batch-size 2 --seq-len 64
"""

import argparse
import os
import sys
from pathlib import Path

import torch
from loguru import logger

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from models.demos.deepseek_v3.reference.configuration_deepseek import DeepseekV3Config
from models.demos.deepseek_v3.utils.config_helpers import sub_state_dict
from models.demos.deepseek_v3.utils.test_utils import dequantize_state_dict, load_state_dict
from models.demos.deepseek_v3_d_p.reference.mla_reference import create_mla_reference


def download_model_weights(cache_dir: Path, layer_idx: int = 0) -> Path:
    """
    Download DeepSeek-R1-0528 model weights from HuggingFace.
    Only downloads necessary files for the specified layer to minimize download size.

    Args:
        cache_dir: Directory to cache downloaded weights
        layer_idx: Which layer to download weights for (default: 0)

    Returns:
        Path to the downloaded model directory
    """
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        logger.error("huggingface_hub is not installed. Install it with: pip install huggingface_hub")
        raise

    model_id = "deepseek-ai/DeepSeek-R1-0528"
    logger.info(f"Downloading DeepSeek-R1-0528 weights from HuggingFace (model: {model_id})")
    logger.info(f"Cache directory: {cache_dir}")
    logger.info(f"Note: Only downloading files needed for layer {layer_idx} to minimize download size")

    # Create cache directory
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Determine which shard file contains layer 0 weights
    # DeepSeek-R1-0528 has weights split across many shards
    # Layer 0 weights are typically in the first few shards
    # We need to download the index file first to know which shard to get

    try:
        # Download essential files + index
        logger.info("Step 1/2: Downloading configuration and index files...")
        allow_patterns = [
            "config.json",
            "*.safetensors.index.json",
            "generation_config.json",
            "tokenizer*",
        ]

        # Add custom model code files (needed for trust_remote_code=True)
        allow_patterns.extend(
            [
                "configuration_deepseek.py",
                "modeling_deepseek.py",
                "*.py",  # Include all Python files for custom model code
            ]
        )

        # First download just the index to figure out which shards we need
        index_dir = snapshot_download(
            repo_id=model_id,
            cache_dir=str(cache_dir),
            allow_patterns=allow_patterns,
            ignore_patterns=["*.safetensors"],  # Don't download weight files yet
        )

        logger.info(f"✓ Configuration downloaded to: {index_dir}")

        # Now download the first few weight shards (layer 0 is usually in first 1-3 shards)
        logger.info("Step 2/2: Downloading weight files for first layer...")
        logger.info("This will download ~3-5GB (first few shards containing layer 0 weights)")

        # Download first 3 shards which should contain layer 0
        shard_patterns = [
            "*-00001-of-*.safetensors",
            "*-00002-of-*.safetensors",
            "*-00003-of-*.safetensors",
        ]

        model_dir = snapshot_download(
            repo_id=model_id,
            cache_dir=str(cache_dir),
            allow_patterns=allow_patterns + shard_patterns,
        )

        logger.success(f"✓ Model weights downloaded successfully!")
        logger.info(f"Model location: {model_dir}")
        return Path(model_dir)

    except Exception as e:
        logger.error(f"Failed to download model: {e}")
        logger.info("You can also manually download the model and use --model-path")
        raise


def get_or_download_model(model_path: Path | None, layer_idx: int = 0) -> Path:
    """
    Get model path, downloading from HuggingFace if necessary.

    Args:
        model_path: User-provided model path (or None)
        layer_idx: Which layer weights to ensure are available

    Returns:
        Path to model directory with weights
    """
    # If user provided a path, use it
    if model_path is not None and model_path.exists():
        # Check if it has the required files
        index_file = model_path / "model.safetensors.index.json"
        if index_file.exists():
            logger.info(f"Using existing model at: {model_path}")
            return model_path
        else:
            logger.warning(f"Model path exists but missing index file: {index_file}")
            logger.info("Will attempt to download from HuggingFace...")

    # Check default location
    default_path = Path("models/demos/deepseek_v3/reference")
    if default_path.exists():
        index_file = default_path / "model.safetensors.index.json"
        if index_file.exists():
            logger.info(f"Using model from default location: {default_path}")
            return default_path

    # Download from HuggingFace
    logger.info("Model not found locally. Downloading DeepSeek-R1-0528 from HuggingFace...")

    # Determine cache directory
    cache_dir = Path(os.getenv("HF_HOME", Path.home() / ".cache" / "huggingface"))
    logger.info(f"Will cache to: {cache_dir}")

    # Ask user for confirmation (can be disabled with --auto-download flag)
    logger.warning("⚠️  This will download ~3-5GB for the first layer weights")
    logger.info("The full DeepSeek-R1-0528 model is large, but we only download what's needed for testing")

    return download_model_weights(cache_dir, layer_idx)


def create_test_config() -> DeepseekV3Config:
    """Create a test configuration for MLA."""
    return DeepseekV3Config(
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


def create_attention_mask(batch_size: int, seq_len: int, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Create causal attention mask."""
    mask = torch.triu(torch.ones(seq_len, seq_len, dtype=dtype) * float("-inf"), diagonal=1)
    return mask.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, seq_len, seq_len)


def create_position_ids(batch_size: int, seq_len: int) -> torch.Tensor:
    """Create position IDs."""
    return torch.arange(seq_len, dtype=torch.long).unsqueeze(0).expand(batch_size, seq_len)


def print_weights_info(state_dict: dict, title: str = "Weights"):
    """
    Print information about weights in state dict.

    Args:
        state_dict: Dictionary of weights
        title: Title for the weight info section
    """
    logger.info("")
    logger.info("=" * 80)
    logger.info(f"{title}")
    logger.info("=" * 80)

    total_params = 0
    total_size_bytes = 0
    for name, tensor in state_dict.items():
        if isinstance(tensor, torch.Tensor):
            num_params = tensor.numel()
            size_bytes = num_params * tensor.element_size()
            size_mb = size_bytes / (1024**2)

            total_params += num_params
            total_size_bytes += size_bytes

            logger.info(
                f"  {name:50s} | shape: {str(tuple(tensor.shape)):20s} | "
                f"params: {num_params:12,d} | size: {size_mb:8.2f} MB | dtype: {tensor.dtype}"
            )

    total_size_mb = total_size_bytes / (1024**2)
    total_size_gb = total_size_bytes / (1024**3)

    logger.info("-" * 80)
    logger.info(f"  Total parameters: {total_params:,d} ({total_params / 1e6:.2f}M)")
    logger.info(f"  Total size: {total_size_mb:.2f} MB ({total_size_gb:.4f} GB)")
    logger.info("=" * 80)
    logger.info("")


def test_with_random_weights(config: DeepseekV3Config, batch_size: int, seq_len: int):
    """
    Test MLA with random weights.

    Args:
        config: Model configuration
        batch_size: Batch size
        seq_len: Sequence length
    """
    logger.info("=" * 80)
    logger.info("Testing MLA with RANDOM WEIGHTS")
    logger.info("=" * 80)

    # Create MLA module with random weights
    logger.info("Creating MLA module with random weights...")
    mla_module = create_mla_reference(
        config=config,
        use_pretrained=False,
        layer_idx=0,
        seed=42,
    )

    # Set to eval mode and bfloat16
    mla_module = mla_module.eval().to(torch.bfloat16)
    logger.info(f"✓ Module created successfully")

    # Print weight information
    print_weights_info(mla_module.state_dict(), "Random Weights (Layer 0)")

    # Create inputs
    logger.info(f"Creating test inputs: batch_size={batch_size}, seq_len={seq_len}, hidden_size={config.hidden_size}")
    hidden_states = torch.randn(batch_size, seq_len, config.hidden_size, dtype=torch.bfloat16)
    attention_mask = create_attention_mask(batch_size, seq_len, dtype=torch.bfloat16)
    position_ids = create_position_ids(batch_size, seq_len)

    # Forward pass
    logger.info("Running forward pass...")
    with torch.no_grad():
        output, attn_weights, cache = mla_module(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=None,
            output_attentions=False,
            use_cache=False,
        )

    # Print results
    logger.info(f"✓ Forward pass successful!")
    logger.info(f"  Input shape:  {hidden_states.shape}")
    logger.info(f"  Output shape: {output.shape}")
    logger.info(f"  Output dtype: {output.dtype}")
    logger.info(f"  Output min:   {output.min().item():.4f}")
    logger.info(f"  Output max:   {output.max().item():.4f}")
    logger.info(f"  Output mean:  {output.mean().item():.4f}")
    logger.info(f"  Output std:   {output.std().item():.4f}")

    # Check for NaN/Inf
    has_nan = torch.isnan(output).any()
    has_inf = torch.isinf(output).any()

    if has_nan or has_inf:
        logger.error(f"✗ Output contains NaN: {has_nan}, Inf: {has_inf}")
        return False
    else:
        logger.success("✓ All checks passed! No NaN or Inf values.")
        return True


def test_with_pretrained_weights(
    config: DeepseekV3Config, model_path: Path | None, batch_size: int, seq_len: int, layer_idx: int = 0
):
    """
    Test MLA with pretrained weights from first layer only.

    Args:
        config: Model configuration (will be replaced with actual config from model)
        model_path: Path to model directory (or None to auto-download)
        batch_size: Batch size
        seq_len: Sequence length
        layer_idx: Which layer to load (default: 0 for first layer only)
    """
    logger.info("=" * 80)
    logger.info(f"Testing MLA with PRETRAINED WEIGHTS (layer {layer_idx} only)")
    logger.info("=" * 80)

    # Get or download model
    model_path = get_or_download_model(model_path, layer_idx)

    # Load actual config from model directory (includes quantization_config)
    logger.info(f"Loading config from: {model_path}")
    from transformers import AutoConfig

    actual_config = AutoConfig.from_pretrained(str(model_path), trust_remote_code=True)
    logger.info("✓ Config loaded from model")

    # Load state dict
    logger.info(f"Loading state dict from: {model_path}")
    state_dict = load_state_dict(model_path, "")
    logger.info("✓ State dict loaded")

    # Extract and dequantize weights for the specific layer
    module_path = f"model.layers.{layer_idx}.self_attn"
    logger.info(f"Extracting weights for module: {module_path}")

    layer_state_dict = sub_state_dict(state_dict, module_path + ".")
    logger.info(f"✓ Found {len(layer_state_dict)} keys for this layer")

    logger.info("Dequantizing weights...")
    dequantized_state_dict = dequantize_state_dict(layer_state_dict, actual_config)
    logger.info("✓ Weights dequantized")

    # Print weight information
    print_weights_info(dequantized_state_dict, f"Pretrained Weights (Layer {layer_idx} - DeepSeek-R1-0528)")

    # Create MLA module with pretrained weights
    logger.info("Creating MLA module with pretrained weights...")
    mla_module = create_mla_reference(
        config=actual_config,
        use_pretrained=True,
        state_dict={module_path + "." + k: v for k, v in dequantized_state_dict.items()},
        layer_idx=layer_idx,
        module_path=module_path,
    )

    # Set to eval mode and bfloat16
    mla_module = mla_module.eval().to(torch.bfloat16)
    logger.info(f"✓ Module created successfully")

    # Create inputs
    logger.info(
        f"Creating test inputs: batch_size={batch_size}, seq_len={seq_len}, hidden_size={actual_config.hidden_size}"
    )
    hidden_states = torch.randn(batch_size, seq_len, actual_config.hidden_size, dtype=torch.bfloat16)
    attention_mask = create_attention_mask(batch_size, seq_len, dtype=torch.bfloat16)
    position_ids = create_position_ids(batch_size, seq_len)

    # Forward pass
    logger.info("Running forward pass...")
    with torch.no_grad():
        output, attn_weights, cache = mla_module(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=None,
            output_attentions=False,
            use_cache=False,
        )

    # Print results
    logger.info(f"✓ Forward pass successful!")
    logger.info(f"  Input shape:  {hidden_states.shape}")
    logger.info(f"  Output shape: {output.shape}")
    logger.info(f"  Output dtype: {output.dtype}")
    logger.info(f"  Output min:   {output.min().item():.4f}")
    logger.info(f"  Output max:   {output.max().item():.4f}")
    logger.info(f"  Output mean:  {output.mean().item():.4f}")
    logger.info(f"  Output std:   {output.std().item():.4f}")

    # Check for NaN/Inf
    has_nan = torch.isnan(output).any()
    has_inf = torch.isinf(output).any()

    if has_nan or has_inf:
        logger.error(f"✗ Output contains NaN: {has_nan}, Inf: {has_inf}")
        return False
    else:
        logger.success("✓ All checks passed! No NaN or Inf values.")
        return True


def main():
    parser = argparse.ArgumentParser(description="Test MLA reference implementation")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["random", "pretrained"],
        default="random",
        help="Test mode: 'random' for random weights, 'pretrained' for downloaded weights",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to pretrained model (optional - will auto-download DeepSeek-R1-0528 from HuggingFace if not provided)",
    )
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for testing")
    parser.add_argument("--seq-len", type=int, default=32, help="Sequence length for testing")
    parser.add_argument("--layer-idx", type=int, default=0, help="Layer index to test (default: 0 for first layer)")

    args = parser.parse_args()

    # Create config
    config = create_test_config()

    # Run tests
    if args.mode == "random":
        success = test_with_random_weights(config, args.batch_size, args.seq_len)
    else:
        # Use provided model path, or None to trigger auto-download
        model_path = Path(args.model_path) if args.model_path else None

        success = test_with_pretrained_weights(config, model_path, args.batch_size, args.seq_len, args.layer_idx)

    if success:
        logger.success("\n" + "=" * 80)
        logger.success("ALL TESTS PASSED ✓")
        logger.success("=" * 80)
        sys.exit(0)
    else:
        logger.error("\n" + "=" * 80)
        logger.error("TESTS FAILED ✗")
        logger.error("=" * 80)
        sys.exit(1)


if __name__ == "__main__":
    main()
