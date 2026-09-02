"""HPSv3++ reward model (Qwen3-VL-8B backbone), loadable in 4-bit on a 12GB GPU.

Background
----------
HPSv3++ (Junjun2333/HPSv3-PlusPlus, arXiv:2606.14657) is a film_hybrid reward
model on top of Qwen3-VL-8B-Instruct. The released checkpoint
(`hpsv3++.pth`, ~17.6GB) is applied via a plain `model.load_state_dict(...,
strict=True)` on a full-precision skeleton -- exactly the same situation as
HPSv3 (Qwen2-VL-7B) in ../hpsv3/src/evaluation/hpsv3_quantized.py,
which this module mirrors. bitsandbytes 4-bit weights have a different
packed shape than the full-precision checkpoint, so quantizing *before*
applying the checkpoint fails with shape mismatches. The fix is the same
two-stage approach:

  1. `merge_and_save_bf16()` -- build the model in bf16 on CPU, apply the
     released checkpoint with `strict=True`, then `save_pretrained()` the
     merged full-precision model to local disk once.
  2. `load_quantized_inferencer()` -- re-load that merged checkpoint through
     `from_pretrained(..., quantization_config=BitsAndBytesConfig(...))`,
     which quantizes weights as they are loaded from disk.

We build our own thin `HPSv3PPQuantizedInferencer` here instead of reusing
the upstream repo's `HPSv3RewardInferencer` / `hpsv3.trainer.adaptive`
plumbing: that plumbing is training-oriented and goes through
`trl.get_quantization_config(ModelConfig)`, whose field names
(`model_args.dtype`, `model_args.bnb_4bit_quant_storage`) don't match the
upstream repo's own `hpsv3.utils.parser.ModelConfig` dataclass under the
trl>=1.10 we have installed (a version-skew bug in the upstream repo, not
something we should paper over by pinning to an old trl). Building the
BitsAndBytesConfig directly, the same way the sibling hpsv3/ project does,
sidesteps that entirely.

The CapabilityEncoder (`cap_encoder`), FiLM generators, and `rm_head` are
excluded from quantization via `llm_int8_skip_modules` -- collectively a few
million parameters, negligible VRAM, and HPSv3++'s own training code keeps
them in fp32 for numerical stability of the conditioning path.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

import torch
from PIL import Image
from transformers import AutoProcessor, BitsAndBytesConfig

# Ensure the upstream HPSv3++ repo (pinned git submodule) and the trl compat
# shim are importable.
_VENDOR_DIR = str(Path(__file__).resolve().parents[2] / "third_party" / "HPSv3-PlusPlus")
if _VENDOR_DIR not in sys.path:
    sys.path.insert(0, _VENDOR_DIR)

from . import _trl_compat  # noqa: F401  (monkey-patches trl.get_kbit_device_map)

from hpsv3.model.qwen3vl_rm import Qwen3VLRewardModelFiLMHybrid  # noqa: E402
from hpsv3.dataset.data_collator_qwen import (  # noqa: E402
    INSTRUCTION,
    prompt_with_special_token,
)
from qwen_vl_utils import process_vision_info  # noqa: E402

# ---------------------------------------------------------------------------
# Constants, from the released checkpoint's config.json / README (model_type
# "film_hybrid", cond_dim 256) and hpsv3/config/train_stage2.yaml defaults.
# ---------------------------------------------------------------------------

BASE_MODEL_NAME = "Qwen/Qwen3-VL-8B-Instruct"
HPSV3PP_REPO_ID = "Junjun2333/HPSv3-PlusPlus"
HPSV3PP_CHECKPOINT_FILE = "hpsv3++.pth"

SPECIAL_REWARD_TOKEN = "<|Reward|>"
OUTPUT_DIM = 2  # rm_head predicts (mu, log_sigma); we select index 0 (mu)
RM_HEAD_TYPE = "ranknet"
RM_HEAD_KWARGS = None
REWARD_TOKEN_MODE = "special"
COND_DIM = 256
MAX_PIXELS = 256 * 28 * 28
MIN_PIXELS = 256 * 28 * 28

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


# ---------------------------------------------------------------------------
# Stage 2: 4-bit quantized reload for inference
# ---------------------------------------------------------------------------


def default_bnb_config(compute_dtype=torch.bfloat16) -> BitsAndBytesConfig:
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,
        llm_int8_skip_modules=["rm_head", "lm_head"] + _FP32_COND_ATTRS,
    )


@dataclass
class HPSv3PPQuantizedInferencer:
    model: Qwen3VLRewardModelFiLMHybrid
    processor: object
    device: str = "cuda"

    @classmethod
    def from_merged_dir(
        cls,
        merged_dir: str,
        device: str = "cuda",
        quantization_config: BitsAndBytesConfig | None = None,
        dtype=torch.bfloat16,
    ) -> "HPSv3PPQuantizedInferencer":
        processor = AutoProcessor.from_pretrained(merged_dir, padding_side="right")
        special_token_ids = processor.tokenizer.convert_tokens_to_ids([SPECIAL_REWARD_TOKEN])

        quantization_config = quantization_config or default_bnb_config(compute_dtype=dtype)

        model = Qwen3VLRewardModelFiLMHybrid.from_pretrained(
            merged_dir,
            output_dim=OUTPUT_DIM,
            reward_token=REWARD_TOKEN_MODE,
            special_token_ids=special_token_ids,
            rm_head_type=RM_HEAD_TYPE,
            rm_head_kwargs=RM_HEAD_KWARGS,
            cond_dim=COND_DIM,
            torch_dtype=dtype,
            attn_implementation="sdpa",
            use_cache=False,
            quantization_config=quantization_config,
            device_map={"": 0},
        )
        model.eval()
        return cls(model=model, processor=processor, device=device)

    def prepare_batch(self, image_paths: Sequence, prompts: Sequence[str]):
        message_list = []
        for text, image in zip(prompts, image_paths):
            out_message = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": image,
                            "min_pixels": MIN_PIXELS,
                            "max_pixels": MAX_PIXELS,
                        },
                        {
                            "type": "text",
                            "text": INSTRUCTION.format(text_prompt=text) + prompt_with_special_token,
                        },
                    ],
                }
            ]
            message_list.append(out_message)

        image_inputs, _ = process_vision_info(message_list)
        batch = self.processor(
            text=self.processor.apply_chat_template(message_list, tokenize=False, add_generation_prompt=True),
            images=image_inputs,
            padding=True,
            return_tensors="pt",
        )
        batch = {k: (v.to(self.device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
        return batch

    @torch.inference_mode()
    def reward(self, image_paths: Sequence, prompts: Sequence[str], iter_step: float = 0.0) -> torch.Tensor:
        """Score (prompt, image) pairs. `iter_step` follows HPSv3++'s
        convention (normalized RL-iteration condition in [0, 1]); 0.0 (the
        pre-RL / initial-model setting) is the recommended default for plain
        preference scoring, per the upstream README."""
        batch = self.prepare_batch(image_paths, prompts)
        bsz = batch["input_ids"].shape[0]
        iter_values = torch.full((bsz,), float(iter_step), dtype=torch.float32, device=self.device)
        out = self.model(return_dict=True, iter_values=iter_values, **batch)["logits"]
        return out

    def score(self, image_paths: Sequence, prompts: Sequence[str], iter_step: float = 0.0) -> List[float]:
        """Return the scalar HPSv3++ score (mu, index 0 of rm_head output)
        for each (image, prompt) pair, matching upstream's `rewards[i][0]`."""
        rewards = self.reward(image_paths, prompts, iter_step=iter_step)
        return [r[0].item() for r in rewards]
