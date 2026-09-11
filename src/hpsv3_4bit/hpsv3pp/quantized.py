"""Inference-only HPSv3++ NF4 runtime."""
from __future__ import annotations
from dataclasses import dataclass
from types import MethodType
from typing import Sequence
import json
from pathlib import Path
import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from qwen_vl_utils import process_vision_info
from safetensors import safe_open
from .hub import load_reward_settings
from .merged_config import load_merged_config
from .model import Qwen3VLRewardModelFiLMHybrid
from .prompts import INSTRUCTION, prompt_with_special_token

SPECIAL_REWARD_TOKEN = "<|Reward|>"
OUTPUT_DIM = 2
RM_HEAD_TYPE = "ranknet"
RM_HEAD_KWARGS = None
COND_DIM = 256
MAX_PIXELS = MIN_PIXELS = 256 * 28 * 28
CAPTION_INSTRUCTION = "Describe this image as a concise text-to-image prompt in one sentence."


def _local_model_dir(directory):
    path = Path(directory).expanduser().resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"Model directory does not exist: {directory}")
    return path


def _load_processor(directory):
    return AutoProcessor.from_pretrained(str(directory), padding_side="right", local_files_only=True, trust_remote_code=False)


def _validate_checkpoint(directory, processor):
    raw = json.loads((directory / "reward_config.json").read_text(encoding="utf-8"))
    token_id = processor.tokenizer.convert_tokens_to_ids(SPECIAL_REWARD_TOKEN)
    defaults = dict(output_dim=OUTPUT_DIM, reward_token="special", special_token_ids=[token_id],
                    rm_head_type=RM_HEAD_TYPE, rm_head_kwargs=RM_HEAD_KWARGS, cond_dim=COND_DIM)
    settings = load_reward_settings(directory, defaults, processor.tokenizer)
    if raw.get("reward_token") != SPECIAL_REWARD_TOKEN or raw.get("reward_token_id") != token_id:
        raise ValueError("Reward token ID does not match the saved tokenizer")
    config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
    quant = config.get("quantization_config") or {}
    if config.get("model_type") != "qwen3_vl" or quant.get("quant_method") != "bitsandbytes" or quant.get("bnb_4bit_quant_type") != "nf4" or not quant.get("load_in_4bit", quant.get("_load_in_4bit", False)):
        raise ValueError("Model is not a serialized bitsandbytes NF4 checkpoint")
    return settings


def _batch(processor, image, text, device):
    messages = [{"role": "user", "content": [{"type": "image", "image": image,
        "min_pixels": MIN_PIXELS, "max_pixels": MAX_PIXELS}, {"type": "text", "text": text}]}]
    image_inputs, _ = process_vision_info(messages)
    batch = processor(text=processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True),
                      images=image_inputs, padding=True, return_tensors="pt")
    return {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}


def _restore_capability_dtype(model, directory):
    # Capability prediction retains the checkpoint precision. Transformers can
    # otherwise preserve the upstream class's FP32 construction dtype.
    dtypes = set()
    for shard in directory.glob("*.safetensors"):
        with safe_open(shard, framework="pt", device="cpu") as weights:
            for key in weights.keys():
                if key.startswith("cap_encoder."):
                    dtypes.add(weights.get_tensor(key).dtype)
    if len(dtypes) != 1 or not dtypes <= {torch.bfloat16, torch.float32}:
        raise ValueError("Capability weights must have one supported floating dtype.")
    model.cap_encoder.to(dtype=dtypes.pop())


@dataclass
class HPSv3PPQuantizedInferencer:
    model: Qwen3VLRewardModelFiLMHybrid
    processor: object
    device: str = "cuda"

    @classmethod
    def from_merged_dir(cls, merged_dir, device="cuda", check_cancel=None):
        check_cancel = check_cancel or (lambda: None)
        directory = _local_model_dir(merged_dir)
        check_cancel()
        processor = _load_processor(directory)
        check_cancel()
        settings = _validate_checkpoint(directory, processor)
        model, info = Qwen3VLRewardModelFiLMHybrid.from_pretrained(
            str(directory), config=load_merged_config(directory, processor.tokenizer), **settings,
            torch_dtype=torch.bfloat16, attn_implementation="sdpa", quantization_config=None,
            device_map={"": str(device)}, use_safetensors=True, output_loading_info=True,
            local_files_only=True, trust_remote_code=False)
        check_cancel()
        if info.get("missing_keys") or info.get("mismatched_keys") or info.get("error_msgs") or info.get("unexpected_keys"):
            raise ValueError(f"Backbone checkpoint mismatch: {info}")
        _restore_capability_dtype(model, directory)
        for name in ("rm_head", "cond_encoder", "film_gen", "scale_gen", "shift_gen", "cond_head", "attn_proj", "margin_head", "sim_proj", "var_proj", "cross_attn", "key_proj", "pair_margin_head", "group_encoder"):
            module = getattr(model, name, None)
            if module is not None:
                module.float()
        model.eval()
        return cls(model, processor, str(device))

    def prepare_batch(self, image_paths: Sequence, prompts: Sequence[str]):
        if len(image_paths) != 1 or len(prompts) != 1:
            raise ValueError("single-pair inference requires exactly one image and prompt")
        batch = _batch(self.processor, image_paths[0], INSTRUCTION.format(text_prompt=prompts[0]) + prompt_with_special_token, self.device)
        token_id = self.processor.tokenizer.convert_tokens_to_ids(SPECIAL_REWARD_TOKEN)
        if (batch["input_ids"] == token_id).sum().item() != 1:
            raise ValueError("Each scoring request must contain exactly one reward token.")
        return batch

    @torch.inference_mode()
    def reward(self, image_paths, prompts, iter_step=0.0):
        batch = self.prepare_batch(image_paths, prompts)
        iteration = torch.full((batch["input_ids"].shape[0],), float(iter_step), dtype=torch.float32, device=self.device)
        return self.model(return_dict=True, iter_values=iteration, **batch)["logits"]

    def score(self, image_paths, prompts):
        rewards = self.reward(image_paths, prompts, iter_step=0.0)
        if rewards.shape[0] != 1:
            raise ValueError("Expected one reward result")
        return [rewards[0, 0].item()]

    @torch.inference_mode()
    def caption(self, image_paths, max_new_tokens=96, instruction=CAPTION_INSTRUCTION, stopping_criteria=None):
        had = "forward" in vars(self.model)
        previous = vars(self.model).get("forward")
        self.model.forward = MethodType(Qwen3VLForConditionalGeneration.forward, self.model)
        try:
            results = []
            for image in image_paths:
                batch = _batch(self.processor, image, instruction, self.device)
                options = dict(**batch, max_new_tokens=max_new_tokens, do_sample=False, use_cache=True,
                               pad_token_id=self.processor.tokenizer.pad_token_id)
                if stopping_criteria is not None:
                    options["stopping_criteria"] = stopping_criteria
                output = self.model.generate(**options)
                results.append(self.processor.tokenizer.decode(output[0, batch["input_ids"].shape[1]:], skip_special_tokens=True).strip())
            return results
        finally:
            if had:
                self.model.forward = previous
            else:
                del self.model.forward
