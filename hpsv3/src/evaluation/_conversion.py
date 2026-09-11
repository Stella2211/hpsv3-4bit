"""BF16 conversion helpers for this scorer release pipeline."""

from __future__ import annotations

import os
import json
from pathlib import Path

import torch
from transformers import AutoProcessor, BitsAndBytesConfig

from .hpsv3_model import Qwen2VLRewardModelBT
from ._hub import load_reward_settings, resolve_model, saved_quantization_config
from ._merged_config import load_merged_config

# Conversion constants from the released HPSv3 training configuration.
BASE_MODEL_NAME = "Qwen/Qwen2-VL-7B-Instruct"
HPSV3_REPO_ID = "MizzenAI/HPSv3"
HPSV3_CHECKPOINT_FILE = "HPSv3.safetensors"
SPECIAL_REWARD_TOKEN = "<|Reward|>"
OUTPUT_DIM = 2
RM_HEAD_TYPE = "ranknet"
RM_HEAD_KWARGS = None
REWARD_TOKEN_MODE = "special"

# Optional BF16 conversion on CPU


def default_bnb_config(compute_dtype=torch.bfloat16):
    return BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype, bnb_4bit_use_double_quant=True,
        llm_int8_skip_modules=["rm_head", "lm_head"],
    )


def export_bnb(merged_dir: str, output_dir: str, device: str = "cuda") -> str:
    """Quantize a BF16 merged checkpoint in the legacy conversion environment."""
    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Output directory must be empty")
    source = resolve_model(merged_dir, local_files_only=True)
    processor = AutoProcessor.from_pretrained(source, padding_side="right", local_files_only=True)
    if SPECIAL_REWARD_TOKEN not in processor.tokenizer.get_vocab():
        processor.tokenizer.add_special_tokens({"additional_special_tokens": [SPECIAL_REWARD_TOKEN]})
    special_ids = processor.tokenizer.convert_tokens_to_ids([SPECIAL_REWARD_TOKEN])
    saved = saved_quantization_config(source)
    if saved is not None and saved.get("quant_method") != "bitsandbytes":
        raise ValueError("This exporter requires bitsandbytes quantization")
    settings = load_reward_settings(source, {
        "output_dim": OUTPUT_DIM, "reward_token": REWARD_TOKEN_MODE,
        "special_token_ids": special_ids, "rm_head_type": RM_HEAD_TYPE,
        "rm_head_kwargs": RM_HEAD_KWARGS,
    }, processor.tokenizer)
    config = load_merged_config(source, processor.tokenizer)
    config.use_cache = False
    model = Qwen2VLRewardModelBT.from_pretrained(
        source, config=config,
        **settings, torch_dtype=torch.bfloat16, attn_implementation="sdpa",
        quantization_config=None if saved is not None else default_bnb_config(),
        device_map={"": device},
    )
    model.save_pretrained(output_dir, safe_serialization=True, max_shard_size="2GB")
    processor.save_pretrained(output_dir)
    # The shared inference loader validates reward settings before loading.
    # Include them in direct exports, not only in prepared Hub releases.
    output.mkdir(parents=True, exist_ok=True)
    (output / "reward_config.json").write_text(json.dumps({
        "format_version": 1, "reward_token": SPECIAL_REWARD_TOKEN,
        "reward_token_id": special_ids[0], "model_kwargs": settings,
    }, indent=2) + "\n", encoding="utf-8")
    return output_dir


def _build_bf16_skeleton(processor) -> Qwen2VLRewardModelBT:
    special_token_ids = processor.tokenizer.convert_tokens_to_ids([SPECIAL_REWARD_TOKEN])
    model = Qwen2VLRewardModelBT.from_pretrained(
        BASE_MODEL_NAME,
        output_dim=OUTPUT_DIM,
        reward_token=REWARD_TOKEN_MODE,
        special_token_ids=special_token_ids,
        rm_head_type=RM_HEAD_TYPE,
        rm_head_kwargs=RM_HEAD_KWARGS,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        use_cache=False,
        low_cpu_mem_usage=True,
    )
    model.resize_token_embeddings(len(processor.tokenizer))
    model.to(torch.bfloat16)
    # HPSv3 keeps the reward head in fp32 even though the backbone is bf16
    # (see hpsv3/train.py: `model.rm_head.to(torch.float32)` after
    # `model.to(torch.bfloat16)`).
    model.rm_head.to(torch.float32)
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    return model


def _apply_checkpoint_streaming(model: Qwen2VLRewardModelBT, checkpoint_path: str) -> None:
    """Apply HPSv3.safetensors to `model` with strict shape checking, without
    loading the whole 16.6GB checkpoint into RAM at once (uses safetensors'
    mmap-backed lazy access, copying one tensor at a time into the already
    allocated bf16/fp32 model parameters)."""
    import safetensors

    # Use state_dict() (not named_parameters()+named_buffers()) so that
    # non-persistent buffers (e.g. rotary embedding caches) are excluded,
    # matching what nn.Module.load_state_dict(strict=True) actually checks
    # against upstream. state_dict() values share storage with the live
    # parameters/buffers (no extra copy), so .copy_() below mutates them.
    own_state = model.state_dict()

    with safetensors.safe_open(checkpoint_path, framework="pt", device="cpu") as f:
        keys = set(f.keys())
        missing = set(own_state.keys()) - keys
        unexpected = keys - set(own_state.keys())
        if missing or unexpected:
            raise RuntimeError(
                f"strict state_dict mismatch: missing={sorted(missing)[:10]} "
                f"(total {len(missing)}), unexpected={sorted(unexpected)[:10]} "
                f"(total {len(unexpected)})"
            )
        for key in keys:
            tensor = f.get_tensor(key)
            target = own_state[key]
            if tensor.shape != target.shape:
                raise RuntimeError(f"shape mismatch for {key}: ckpt={tensor.shape} model={target.shape}")
            with torch.no_grad():
                target.data.copy_(tensor.to(target.dtype))
            del tensor


def merge_and_save_bf16(output_dir: str, checkpoint_path: str | None = None, force: bool = False) -> str:
    """Build HPSv3 in bf16 on CPU, apply the fine-tuned checkpoint, and save
    the merged full model + processor to `output_dir`. Returns output_dir.

    This step needs no GPU. Peak host RAM is roughly one bf16 7B model
    (~16GB) since the checkpoint is applied tensor-by-tensor via mmap rather
    than loaded wholesale.
    """
    output_dir = str(output_dir)
    done_marker = Path(output_dir) / "MERGE_COMPLETE"
    if done_marker.exists() and not force:
        return output_dir

    os.makedirs(output_dir, exist_ok=True)

    processor = AutoProcessor.from_pretrained(BASE_MODEL_NAME, padding_side="right")
    processor.tokenizer.add_special_tokens({"additional_special_tokens": [SPECIAL_REWARD_TOKEN]})

    model = _build_bf16_skeleton(processor)

    if checkpoint_path is None:
        import huggingface_hub

        checkpoint_path = huggingface_hub.hf_hub_download(
            HPSV3_REPO_ID, HPSV3_CHECKPOINT_FILE, repo_type="model"
        )

    _apply_checkpoint_streaming(model, checkpoint_path)
    model.eval()

    model.save_pretrained(output_dir, safe_serialization=True)
    processor.save_pretrained(output_dir)
    done_marker.write_text("ok\n")
    return output_dir
