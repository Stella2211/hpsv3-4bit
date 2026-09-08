"""HPSv3 reward scoring with bitsandbytes NF4.

HPSv3QuantizedInferencer.from_merged_dir() downloads the published
NF4 model by default. Local checkpoints and Hub repository IDs are also
accepted; serialized NF4 settings are reused when present.

Conversion helpers build a BF16 model on CPU and apply the reward checkpoint
before quantization, since packed 4-bit weights cannot accept a full-precision
state dict. This conversion is unnecessary for the published NF4 models.
Reward and conditioning modules are excluded from 4-bit quantization;
exclusion alone does not preserve FP32 dtype.
"""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from types import MethodType
from typing import List, Sequence

import torch
from PIL import Image
from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2VLForConditionalGeneration
from qwen_vl_utils import process_vision_info
from ._hub import resolve_model, load_reward_settings, saved_quantization_config

from .hpsv3_model import Qwen2VLRewardModelBT
from ._merged_config import load_merged_config

# ---------------------------------------------------------------------------
# Constants copied verbatim from HPSv3's hpsv3/dataset/data_collator_qwen.py
# and hpsv3/config/HPSv3_7B.yaml (the released checkpoint's training config).
# ---------------------------------------------------------------------------

BASE_MODEL_NAME = "Qwen/Qwen2-VL-7B-Instruct"
HPSV3_REPO_ID = "MizzenAI/HPSv3"
HPSV3_CHECKPOINT_FILE = "HPSv3.safetensors"

SPECIAL_REWARD_TOKEN = "<|Reward|>"
OUTPUT_DIM = 2  # rm_head predicts (mu, log_sigma); HPSv3 selects index 0 (mu)
RM_HEAD_TYPE = "ranknet"
RM_HEAD_KWARGS = None  # HPSv3_7B.yaml doesn't set this -> default 3-layer MLP
REWARD_TOKEN_MODE = "special"
MAX_PIXELS = 256 * 28 * 28
MIN_PIXELS = 256 * 28 * 28

INSTRUCTION = """
You are tasked with evaluating a generated image based on Visual Quality and Text Alignment and give a overall score to estimate the human preference. Please provide a rating from 0 to 10, with 0 being the worst and 10 being the best.

**Visual Quality:**
Evaluate the overall visual quality of the image. The following sub-dimensions should be considered:
- **Reasonableness:** The image should not contain any significant biological or logical errors, such as abnormal body structures or nonsensical environmental setups.
- **Clarity:** Evaluate the sharpness and visibility of the image. The image should be clear and easy to interpret, with no blurring or indistinct areas.
- **Detail Richness:** Consider the level of detail in textures, materials, lighting, and other visual elements (e.g., hair, clothing, shadows).
- **Aesthetic and Creativity:** Assess the artistic aspects of the image, including the color scheme, composition, atmosphere, depth of field, and the overall creative appeal. The scene should convey a sense of harmony and balance.
- **Safety:** The image should not contain harmful or inappropriate content, such as political, violent, or adult material. If such content is present, the image quality and satisfaction score should be the lowest possible.

**Text Alignment:**
Assess how well the image matches the textual prompt across the following sub-dimensions:
- **Subject Relevance** Evaluate how accurately the subject(s) in the image (e.g., person, animal, object) align with the textual description. The subject should match the description in terms of number, appearance, and behavior.
- **Style Relevance:** If the prompt specifies a particular artistic or stylistic style, evaluate how well the image adheres to this style.
- **Contextual Consistency**: Assess whether the background, setting, and surrounding elements in the image logically fit the scenario described in the prompt. The environment should support and enhance the subject without contradictions.
- **Attribute Fidelity**: Check if specific attributes mentioned in the prompt (e.g., colors, clothing, accessories, expressions, actions) are faithfully represented in the image. Minor deviations may be acceptable, but critical attributes should be preserved.
- **Semantic Coherence**: Evaluate whether the overall meaning and intent of the prompt are captured in the image. The generated content should not introduce elements that conflict with or distort the original description.
Textual prompt - {text_prompt}


"""

PROMPT_WITH_SPECIAL_TOKEN = """
Please provide the overall ratings of this image: <|Reward|>

END
"""

CAPTION_INSTRUCTION = "Describe this image as a concise text-to-image prompt in one sentence."


# ---------------------------------------------------------------------------
# Optional BF16 conversion on CPU
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# NF4 loading and inference
# ---------------------------------------------------------------------------


def default_bnb_config(compute_dtype=torch.bfloat16) -> BitsAndBytesConfig:
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,
        llm_int8_skip_modules=["rm_head", "lm_head"],
    )


def _load_processor(merged_dir: str, processor_dir: str | None, base_model_name: str, local_files_only=False):
    """Load the AutoProcessor, preferring: explicit `processor_dir` >
    `merged_dir` (if it actually contains processor files) > the base model
    id. Merged dirs produced by `merge_and_save_bf16()` include the
    processor; a weights-only export (config.json + safetensors) does not,
    in which case we fall back to the base model's processor and re-add the
    reward token exactly like the merge step does (appended at the end of
    the vocab, matching the merged model's resized embeddings)."""
    if processor_dir is not None:
        source = processor_dir
    elif (Path(merged_dir) / "preprocessor_config.json").exists():
        source = merged_dir
    else:
        source = base_model_name
    processor = AutoProcessor.from_pretrained(source, padding_side="right", local_files_only=local_files_only)
    # Transformers 4.46 predates processor chat_template.jinja loading.
    template = Path(source) / "chat_template.jinja"
    if not processor.chat_template and template.is_file():
        processor.chat_template = template.read_text(encoding="utf-8")
    if not processor.chat_template:
        processor.chat_template = processor.tokenizer.chat_template
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
    """Warn when auxiliary modules differ from the upstream FP32 setup.

    Excluding modules from quantization does not prevent dtype conversion.
    This diagnostic permits checkpoints with BF16 auxiliary modules.
    """
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
class HPSv3QuantizedInferencer:
    model: Qwen2VLRewardModelBT
    processor: object
    device: str = "cuda"

    @classmethod
    def from_merged_dir(
        cls,
        merged_dir: str | None = None,
        device: str = "cuda",
        quantization_config: BitsAndBytesConfig | None = None,
        dtype=torch.bfloat16,
        processor_dir: str | None = None,
        revision: str | None = None,
        local_files_only: bool = False,
    ) -> "HPSv3QuantizedInferencer":
        merged_dir = resolve_model(merged_dir, revision=revision, local_files_only=local_files_only)
        processor = _load_processor(merged_dir, processor_dir, BASE_MODEL_NAME, local_files_only)
        special_token_ids = processor.tokenizer.convert_tokens_to_ids([SPECIAL_REWARD_TOKEN])

        saved_quant = saved_quantization_config(merged_dir)
        if saved_quant is not None:
            if quantization_config is not None:
                raise ValueError('A prequantized model uses its saved quantization settings')
            if saved_quant.get("quant_method") != "bitsandbytes":
                raise ValueError("This loader requires bitsandbytes quantization")
            # Transformers reads the serialized settings directly from config.
            quantization_config = None
        else:
            quantization_config = quantization_config or default_bnb_config(compute_dtype=dtype)
        reward_settings = load_reward_settings(merged_dir, dict(
            output_dim=OUTPUT_DIM, reward_token=REWARD_TOKEN_MODE,
            special_token_ids=special_token_ids, rm_head_type=RM_HEAD_TYPE,
            rm_head_kwargs=RM_HEAD_KWARGS,
        ), processor.tokenizer)

        config = load_merged_config(merged_dir, processor.tokenizer)
        config.use_cache = False
        model = Qwen2VLRewardModelBT.from_pretrained(
            merged_dir,
            config=config,
            **reward_settings,
            torch_dtype=dtype,
            attn_implementation="sdpa",
            quantization_config=quantization_config,
            device_map=_device_map_from_device(device),
        )
        model.eval()
        _warn_if_not_fp32(model, ["rm_head"])
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
                            "text": INSTRUCTION.format(text_prompt=text) + PROMPT_WITH_SPECIAL_TOKEN,
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
    def reward(self, image_paths: Sequence, prompts: Sequence[str]) -> torch.Tensor:
        batch = self.prepare_batch(image_paths, prompts)
        out = self.model(return_dict=True, **batch)["logits"]
        return out

    def score(self, image_paths: Sequence, prompts: Sequence[str]) -> List[float]:
        """Return the scalar HPSv3 score (mu, index 0 of rm_head output) for
        each (image, prompt) pair, matching HPSv3's own inference.py which
        does `rewards[i][0].item()`."""
        rewards = self.reward(image_paths, prompts)
        return [r[0].item() for r in rewards]

    @torch.inference_mode()
    def caption(
        self,
        image_paths: Sequence,
        max_new_tokens: int = 96,
        instruction: str = CAPTION_INSTRUCTION,
    ) -> List[str]:
        """Generate a short text-to-image-style caption for each image using
        the (4-bit) Qwen2-VL backbone itself, so images without prompts can
        still be scored (see score_batch.py --no-prompt).

        The merged HPSv3 checkpoint carries the full ``lm_head`` weights
        (the merge step verifies the checkpoint against the complete state
        dict), so the language-model path is real fine-tuned/base weights,
        not a random init. ``Qwen2VLRewardModelBT`` overrides ``forward()``
        to return pooled reward logits, which would break ``generate()``;
        the base-class forward is temporarily rebound for the duration of
        the call, leaving the scoring path untouched.

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
        self.model.forward = MethodType(Qwen2VLForConditionalGeneration.forward, self.model)
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
