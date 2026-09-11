"""BF16 conversion helpers for this scorer release pipeline."""

from __future__ import annotations

import os
import json
import sys
from pathlib import Path

import torch
from transformers import AutoProcessor, BitsAndBytesConfig

# Ensure the upstream HPSv3++ repo (pinned git submodule) and the trl compat
# shim are importable.
_VENDOR_DIR = str(Path(__file__).resolve().parents[2] / "third_party" / "HPSv3-PlusPlus")
if _VENDOR_DIR not in sys.path:
    sys.path.insert(0, _VENDOR_DIR)


from . import _trl_compat  # noqa: F401

from hpsv3.model.qwen3vl_rm import Qwen3VLRewardModelFiLMHybrid  # noqa: E402
from ._hub import load_reward_settings, resolve_model, saved_quantization_config
from ._merged_config import load_merged_config

# Conversion constants from the released checkpoint configuration.
BASE_MODEL_NAME = "Qwen/Qwen3-VL-8B-Instruct"
HPSV3PP_REPO_ID = "Junjun2333/HPSv3-PlusPlus"
HPSV3PP_CHECKPOINT_FILE = "hpsv3++.pth"
SPECIAL_REWARD_TOKEN = "<|Reward|>"
OUTPUT_DIM = 2
RM_HEAD_TYPE = "ranknet"
RM_HEAD_KWARGS = None
REWARD_TOKEN_MODE = "special"
COND_DIM = 256

# Condition-related submodules that hpsv3/trainer/adaptive.py keeps in fp32
# (see _create_model_common); we mirror that for the merge step, and also
# use it as the bitsandbytes int8/4bit skip list (tiny, and precision here
# matters more than VRAM).
_FP32_COND_ATTRS = [
    "level_embedding", "iter_proj", "film_gen", "cond_encoder",
    "scale_gen", "shift_gen", "cond_head", "attn_proj", "margin_head",
    "sim_proj", "var_proj", "cross_attn", "key_proj",
    "pair_margin_head", "group_encoder", "cap_encoder",
]


def default_bnb_config(compute_dtype=torch.bfloat16):
    return BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype, bnb_4bit_use_double_quant=True,
        llm_int8_skip_modules=["rm_head", "lm_head", *_FP32_COND_ATTRS],
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
        "rm_head_kwargs": RM_HEAD_KWARGS, "cond_dim": COND_DIM,
    }, processor.tokenizer)
    config = load_merged_config(source, processor.tokenizer)
    config.use_cache = False
    model = Qwen3VLRewardModelFiLMHybrid.from_pretrained(
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


def _build_bf16_skeleton(processor, base_model_name: str) -> Qwen3VLRewardModelFiLMHybrid:
    special_token_ids = processor.tokenizer.convert_tokens_to_ids([SPECIAL_REWARD_TOKEN])
    model = Qwen3VLRewardModelFiLMHybrid.from_pretrained(
        base_model_name,
        output_dim=OUTPUT_DIM,
        reward_token=REWARD_TOKEN_MODE,
        special_token_ids=special_token_ids,
        rm_head_type=RM_HEAD_TYPE,
        rm_head_kwargs=RM_HEAD_KWARGS,
        cond_dim=COND_DIM,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        use_cache=False,
        low_cpu_mem_usage=True,
    )
    model.resize_token_embeddings(len(processor.tokenizer))
    model.to(torch.bfloat16)
    model.rm_head.to(torch.float32)
    for attr in _FP32_COND_ATTRS:
        if hasattr(model, attr):
            getattr(model, attr).to(torch.float32)
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    return model


def _apply_checkpoint(model: Qwen3VLRewardModelFiLMHybrid, checkpoint_path: str) -> None:
    """Apply hpsv3++.pth to `model` with strict shape checking. The
    checkpoint is a plain torch .pth (not safetensors), so unlike the
    sibling hpsv3/ project's safetensors-based streaming trick we load it
    via torch.load."""
    # mmap=True avoids materializing the whole ~17.6GB pickle in RAM at once
    # (it has to coexist with a bf16 8B skeleton of similar size, so this
    # keeps peak host RAM at roughly ~40GB instead of ~55GB).
    state_dict = torch.load(checkpoint_path, map_location="cpu", mmap=True)
    if isinstance(state_dict, dict) and "model" in state_dict:
        state_dict = state_dict["model"]
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    # strict=False + manual check (rather than strict=True) so we can report
    # a readable diff instead of transformers' truncated default message.
    if missing or unexpected:
        raise RuntimeError(
            f"state_dict mismatch: missing={missing[:10]} (total {len(missing)}), "
            f"unexpected={unexpected[:10]} (total {len(unexpected)})"
        )


def merge_and_save_bf16(
    output_dir: str,
    checkpoint_path: str | None = None,
    base_model_name: str = BASE_MODEL_NAME,
    force: bool = False,
) -> str:
    """Build HPSv3++ in bf16 on CPU, apply the released checkpoint, and save
    the merged full model + processor to `output_dir`. Returns output_dir.
    Needs no GPU."""
    output_dir = str(output_dir)
    done_marker = Path(output_dir) / "MERGE_COMPLETE"
    if done_marker.exists() and not force:
        return output_dir

    os.makedirs(output_dir, exist_ok=True)

    processor = AutoProcessor.from_pretrained(base_model_name, padding_side="right")
    processor.tokenizer.add_special_tokens({"additional_special_tokens": [SPECIAL_REWARD_TOKEN]})

    model = _build_bf16_skeleton(processor, base_model_name)

    if checkpoint_path is None:
        import huggingface_hub

        checkpoint_path = huggingface_hub.hf_hub_download(
            HPSV3PP_REPO_ID, HPSV3PP_CHECKPOINT_FILE, repo_type="model"
        )

    _apply_checkpoint(model, checkpoint_path)
    model.eval()

    model.save_pretrained(output_dir, safe_serialization=True)
    processor.save_pretrained(output_dir)
    done_marker.write_text("ok\n")
    return output_dir
