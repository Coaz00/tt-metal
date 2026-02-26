# Objective: Take a GPT2 model that can complete sentences, and then train it to solve math problems from GSM8K.
# Currently the reward is the negative answer length, meaning we don't prioritize correct answers, just short answers.

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import hf_hub_download
from ttml.common.model_factory import TransformerModelFactory
import ttnn
import ttml
import os
import numpy as np

CONFIG = "training_gsm8k_rl_gpt2.yaml"

from ttml.common.config import (
    TrainingConfig,
    DeviceConfig,
    SchedulerConfig,
    load_config,
    yaml_deep_update,
)

from ttml.common.utils import (
    # round_up_to_tile,
    # initialize_device,
    # create_optimizer,
    # get_loss_over_devices,
    # build_logits_mask,
    # no_grad,
    get_tt_metal_home,
)


# gsm8k_dataset = load_dataset("openai/gsm8k", "main")
# training_data = gsm8k_dataset["train"]
# testing_data = gsm8k_dataset["test"]


def model_inference(tt_model, prompt):
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    input_ids = tokenizer.encode(prompt)
    generated_ids = list(input_ids)

    MAX_T = 1024
    assert len(generated_ids) < MAX_T, "Prompt too long for max_sequence_length"

    for _ in range(20):
        actual_len = len(generated_ids)

        # ---- input ids: [1,1,1,MAX_T], uint32, ROW_MAJOR ----
        curr_ids = np.array(generated_ids, dtype=np.uint32)
        curr_ids = np.pad(
            curr_ids, (0, MAX_T - actual_len), mode="constant", constant_values=0
        )

        input_tensor = ttml.autograd.Tensor.from_numpy(
            curr_ids.reshape(1, 1, 1, MAX_T),
            layout=ttnn.Layout.ROW_MAJOR,
            new_type=ttnn.DataType.UINT32,
        )

        # ---- causal mask: [1,1,MAX_T,MAX_T], bf16, TILE ----
        mask_np = np.tril(np.ones((actual_len, actual_len), dtype=np.float32))
        mask_np = np.pad(
            mask_np,
            ((0, MAX_T - actual_len), (0, MAX_T - actual_len)),
            mode="constant",
            constant_values=0,
        )

        mask = ttml.autograd.Tensor.from_numpy(
            mask_np.reshape(1, 1, MAX_T, MAX_T),
            layout=ttnn.Layout.TILE,
            new_type=ttnn.DataType.BFLOAT16,
        )

        # ---- forward ----
        logits = tt_model(input_tensor, mask)
        logits_np = logits.to_numpy(ttnn.DataType.FLOAT32)  # [1,1,MAX_T,padded_vocab]

        # read logits at last REAL token position, and clip vocab to tokenizer vocab
        vocab_size = tokenizer.vocab_size
        logits = logits_np[0, 0, actual_len - 1, :vocab_size]

        temperature = 0.8
        top_k = 50

        scaled = logits / temperature
        top_idx = np.argpartition(scaled, -top_k)[-top_k:]
        top_logits = scaled[top_idx]
        probs = np.exp(top_logits - np.max(top_logits))
        probs = probs / probs.sum()

        next_token_id = int(np.random.choice(top_idx, p=probs))

        # next_token_id = int(np.argmax(last_token_logits))

        generated_ids.append(next_token_id)

        # optional debug
        # print("next_token_id:", next_token_id, "token:", repr(tokenizer.decode([next_token_id])))

    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=False)
    print(f"\nPrompt: {prompt}")
    print(f"Generated: {generated_text}")


def load_training_config():
    yaml_config = load_config(
        CONFIG, f"{get_tt_metal_home()}/tt-train/configs/training_configs"
    )

    print(f"YAML config: {yaml_config}")
    model_config = load_config(yaml_config["training_config"]["model_config"])

    override_config_path = (
        f"{os.environ['TT_METAL_HOME']}/tt-train/configs/training_overrides.yaml"
    )

    if os.path.isfile(override_config_path):
        print("Applying training overrides...")

        override_config = load_config(override_config_path)

        yaml_config = yaml_deep_update(yaml_config, override_config)
        model_config = yaml_deep_update(model_config, override_config)

        # pretty output of yaml config
        import yaml

        print("Loaded YAML config:")
        print(yaml.dump(yaml_config, sort_keys=False, default_flow_style=False))
        print("*********************************\n\n")

    return model_config


def create_model(model_config):
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    safetensors_path = hf_hub_download(
        repo_id="gpt2",
        filename="model.safetensors",
    )

    safetensors_path = safetensors_path.replace("model.safetensors", "")

    print(f"Safetensors path: {safetensors_path}")

    # Setup model
    print("Setting up model...")
    orig_vocab_size = tokenizer.vocab_size
    tt_model_factory = TransformerModelFactory(model_config)
    tt_model_factory.transformer_config.vocab_size = orig_vocab_size
    print("Created Model Factory")

    print("Creating model...")
    tt_model = tt_model_factory.create_model()
    print("Loading from safetensors...")
    tt_model.load_from_safetensors(safetensors_path)
    return tt_model


if __name__ == "__main__":
    model_config = load_training_config()
    print(model_config)

    tt_model = create_model(model_config)
    print(tt_model.__dir__())

    prompt = "The capital of France is"

    model_inference(tt_model, prompt)
