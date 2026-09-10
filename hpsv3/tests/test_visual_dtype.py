import sys
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "hpsv3/src"))

from evaluation import hpsv3_quantized as quantized


class _Processor:
    class _Tokenizer:
        pad_token_id = 0

        def convert_tokens_to_ids(self, tokens):
            return [1]

        def get_vocab(self):
            return {quantized.SPECIAL_REWARD_TOKEN: 1}

        def decode(self, ids, skip_special_tokens=True):
            return "a caption"

    tokenizer = _Tokenizer()

    def apply_chat_template(self, messages, **kwargs):
        return "prompt"

    def __call__(self, **kwargs):
        return {
            "input_ids": torch.tensor([[1]]),
            "attention_mask": torch.tensor([[1]]),
            "pixel_values": torch.tensor([-1.5, 0.25, 0.75]),
            "image_grid_thw": torch.tensor([[1, 1, 2]]),
        }


class _Visual:
    def __init__(self):
        self.patch_embed = SimpleNamespace(proj=SimpleNamespace(weight=torch.empty((), dtype=torch.bfloat16)))
        self.blocks = [SimpleNamespace(mlp=SimpleNamespace(fc2=SimpleNamespace(
            weight=torch.zeros(1, dtype=torch.uint8)
        )))]

    def get_dtype(self):
        return self.blocks[0].mlp.fc2.weight.dtype


class _Model:
    def __init__(self):
        self.visual = _Visual()
        self.seen = []

    def eval(self):
        return self

    def __call__(self, **batch):
        self.seen.append(batch["pixel_values"].to(self.visual.get_dtype()))
        return {"logits": torch.tensor([[1.0, 0.0]])}

    def generate(self, **batch):
        self.forward(pixel_values=batch["pixel_values"])
        return torch.tensor([[1, 2]])


def _base_forward(self, pixel_values=None, **kwargs):
    self.seen.append(pixel_values.to(self.visual.get_dtype()))


class VisualDtypeTests(unittest.TestCase):
    def test_patch_uses_floating_patch_projection_dtype_without_changing_weights(self):
        model = _Model()
        weight = model.visual.patch_embed.proj.weight
        packed_weight = model.visual.blocks[0].mlp.fc2.weight
        self.assertEqual(model.visual.get_dtype(), torch.uint8)

        quantized._patch_quantized_visual_dtype(model)

        self.assertEqual(model.visual.get_dtype(), torch.bfloat16)
        self.assertIs(model.visual.patch_embed.proj.weight, weight)
        self.assertEqual(weight.dtype, torch.bfloat16)
        self.assertIs(model.visual.blocks[0].mlp.fc2.weight, packed_weight)
        self.assertEqual(packed_weight.dtype, torch.uint8)

    def test_loader_installs_instance_override(self):
        model = _Model()
        processor = _Processor()

        with patch.object(quantized, "resolve_model", return_value="merged"), \
                patch.object(quantized, "_load_processor", return_value=processor), \
                patch.object(quantized, "saved_quantization_config", return_value={"quant_method": "bitsandbytes"}), \
                patch.object(quantized, "load_reward_settings", return_value={}), \
                patch.object(quantized, "load_merged_config", return_value=SimpleNamespace(use_cache=True)), \
                patch.object(quantized.Qwen2VLRewardModelBT, "from_pretrained", return_value=model):
            inferencer = quantized.HPSv3QuantizedInferencer.from_merged_dir(
                "merged", device="cpu", local_files_only=True
            )

        self.assertEqual(inferencer.model.visual.get_dtype(), torch.bfloat16)

    def test_score_and_caption_preserve_fractional_pixel_signal(self):
        model = _Model()
        inferencer = quantized.HPSv3QuantizedInferencer(model, _Processor(), device="cpu")
        quantized._patch_quantized_visual_dtype(model)

        with patch.object(quantized, "process_vision_info", return_value=([], None)), \
                patch.object(quantized.Qwen2VLForConditionalGeneration, "forward", _base_forward):
            inferencer.score(["image"], ["prompt"])
            inferencer.caption(["image"])

        self.assertEqual(len(model.seen), 2)
        self.assertTrue(all(values.dtype == torch.bfloat16 for values in model.seen))
        expected = torch.tensor([-1.5, 0.25, 0.75], dtype=torch.bfloat16)
        for values in model.seen:
            torch.testing.assert_close(values, expected)
