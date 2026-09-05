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
import warnings
from dataclasses import dataclass
from pathlib import Path
from types import MethodType
from typing import List, Sequence

import torch
from PIL import Image
from transformers import AutoProcessor, BitsAndBytesConfig, Qwen3VLForConditionalGeneration

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

CAPTION_INSTRUCTION = "Describe this image as a concise text-to-image prompt in one sentence."

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


def _load_processor(merged_dir: str, processor_dir: str | None, base_model_name: str):
    """Load the AutoProcessor, preferring: explicit `processor_dir` >
    `merged_dir` (if it actually contains processor files) > the base model
    id. Merged dirs produced by `merge_and_save_bf16()` include the
    processor; the community pre-merged bdsqlsz/HPSV3-PlusPLus-BF16 export
    ships only config.json + weight safetensors, in which case we fall back
    to the base model's processor and re-add the reward token exactly like
    the merge step does (appended at the end of the vocab, matching the
    merged model's resized embeddings)."""
    if processor_dir is not None:
        source = processor_dir
    elif (Path(merged_dir) / "preprocessor_config.json").exists():
        source = merged_dir
    else:
        source = base_model_name
    processor = AutoProcessor.from_pretrained(source, padding_side="right")
    if SPECIAL_REWARD_TOKEN not in processor.tokenizer.get_vocab():
        processor.tokenizer.add_special_tokens(
            {"additional_special_tokens": [SPECIAL_REWARD_TOKEN]}
        )
    return processor


def _device_map_from_device(device: str):
    """Derive a transformers `device_map` from a torch-style device string.
    "cuda" (no index) keeps the historical default {"": 0}; "cuda:N" pins
    all modules to GPU N."""
    if device.startswith("cuda"):
        index = int(device.split(":", 1)[1]) if ":" in device else 0
        return {"": index}
    return {"": device}


def _warn_if_not_fp32(model, module_names: Sequence[str]) -> None:
    """Lightweight post-load sanity check: these modules are in
    `llm_int8_skip_modules` and were saved in fp32, so they should still be
    fp32 after the quantized reload. Warn (not assert) so that custom
    quantization configs remain usable."""
    for name in module_names:
        module = getattr(model, name, None)
        if module is None:
            continue
        bad = {str(p.dtype) for p in module.parameters() if p.dtype != torch.float32}
        if bad:
            warnings.warn(
                f"expected model.{name} to be fp32 after load, found {sorted(bad)}; "
                "scores may not match the reference setup"
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
        processor_dir: str | None = None,
    ) -> "HPSv3PPQuantizedInferencer":
        processor = _load_processor(merged_dir, processor_dir, BASE_MODEL_NAME)
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
            device_map=_device_map_from_device(device),
        )
        model.eval()
        _warn_if_not_fp32(model, ["rm_head", *_FP32_COND_ATTRS])
        return cls(model=model, processor=processor, device=device)

    def prepare_batch(self, image_paths: Sequence, prompts: Sequence[str]):
        if len(image_paths) != len(prompts):
            raise ValueError(
                f"got {len(image_paths)} images but {len(prompts)} prompts; "
                "they must pair up 1:1"
            )
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

    @torch.inference_mode()
    def caption(
        self,
        image_paths: Sequence,
        max_new_tokens: int = 96,
        instruction: str = CAPTION_INSTRUCTION,
    ) -> List[str]:
        """Generate a short text-to-image-style caption for each image using
        the (4-bit) Qwen3-VL backbone itself, so images without prompts can
        still be scored (see score_batch.py --no-prompt).

        The merged HPSv3++ checkpoint carries the full ``lm_head`` weights
        (the merge step verifies the checkpoint against the complete state
        dict), so the language-model path is real weights, not a random
        init. The reward subclass overrides ``forward()`` to return pooled
        reward logits, which would break ``generate()``; the base-class
        forward is temporarily rebound for the duration of the call,
        leaving the scoring path untouched.

        Images are processed one at a time: the processor is configured with
        ``padding_side="right"`` for reward scoring, which is the wrong
        padding side for batched generation.
        """
        if getattr(self, "_captioning", False):
            raise RuntimeError("caption() is not reentrant: the model's forward() is temporarily rebound")
        self._captioning = True
        # If something (e.g. Accelerate) already set an instance-level
        # forward, keep it around and put it back afterwards instead of
        # unconditionally deleting.
        had_instance_forward = "forward" in vars(self.model)
        prev_forward = vars(self.model).get("forward")
        captions: List[str] = []
        self.model.forward = MethodType(Qwen3VLForConditionalGeneration.forward, self.model)
        try:
            for image in image_paths:
                messages = [
                    [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image",
                                    "image": image,
                                    "min_pixels": MIN_PIXELS,
                                    "max_pixels": MAX_PIXELS,
                                },
                                {"type": "text", "text": instruction},
                            ],
                        }
                    ]
                ]
                image_inputs, _ = process_vision_info(messages)
                batch = self.processor(
                    text=self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True),
                    images=image_inputs,
                    padding=True,
                    return_tensors="pt",
                )
                batch = {k: (v.to(self.device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
                input_len = batch["input_ids"].shape[1]
                out = self.model.generate(
                    **batch,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    use_cache=True,
                    pad_token_id=self.processor.tokenizer.pad_token_id,
                )
                text = self.processor.tokenizer.decode(out[0][input_len:], skip_special_tokens=True).strip()
                captions.append(text)
        finally:
            # Restore the reward-model forward.
            if had_instance_forward:
                self.model.forward = prev_forward
            else:
                del self.model.forward
            self._captioning = False
        return captions
