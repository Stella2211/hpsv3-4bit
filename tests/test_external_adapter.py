import sys
import types
import unittest

import torch


class _ExternalBase(torch.nn.Module):
    def __init__(self):
        super().__init__()


class _Visual(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.num_grid_per_side = 2
        self.interpolation_mode = "bilinear"
        self.interpolation_align_corners = False
        self.spatial_merge_size = 1
        self.pos_embed = torch.nn.Embedding(4, 2, dtype=torch.bfloat16)
        self.seen = None

    def forward(self, hidden, grid_thw, **kwargs):
        self.seen = kwargs
        return hidden


class _Backbone(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.visual = _Visual()
        self.seen = None

    def forward(self, **kwargs):
        self.seen = kwargs
        return self.visual(torch.zeros(4, 2), kwargs.get("image_grid_thw"), **kwargs)


class ExternalAdapterTests(unittest.TestCase):
    def setUp(self):
        self.old = sys.modules.get("hpsv3_4bit.hpsv3pp.upstream")
        fake = types.ModuleType("hpsv3_4bit.hpsv3pp.upstream")
        fake.load_model_module = lambda: types.SimpleNamespace(
            Qwen3VLRewardModelBT=_ExternalBase,
            Qwen3VLRewardModelFiLMContinuous=_ExternalBase,
            Qwen3VLRewardModelFiLMHybrid=_ExternalBase,
        )
        fake.load_prompts = lambda: {"INSTRUCTION": "{text_prompt}", "prompt_with_special_token": "<R>"}
        sys.modules[fake.__name__] = fake

    def tearDown(self):
        if self.old is None:
            sys.modules.pop("hpsv3_4bit.hpsv3pp.upstream", None)
        else:
            sys.modules["hpsv3_4bit.hpsv3pp.upstream"] = self.old

    def test_class_is_lazy_thin_adapter(self):
        from hpsv3_4bit.hpsv3pp.model import get_reward_model_class

        cls = get_reward_model_class()
        self.assertTrue(issubclass(cls, _ExternalBase))
        self.assertEqual(cls._keep_in_fp32_modules_strict, ["rm_head", "cond_encoder", "film_gen"])

    def test_visual_hook_casts_interpolation_weights_to_position_dtype(self):
        from hpsv3_4bit.hpsv3pp.model import install_vision_interpolation_hook

        model = types.SimpleNamespace(model=_Backbone(), rm_head=torch.nn.Linear(2, 1))
        handle = install_vision_interpolation_hook(model)
        try:
            grid = torch.tensor([[1, 2, 2]])
            model.model.visual(torch.zeros(4, 2), grid)
            self.assertEqual(model.model.visual.seen["interp_weights"].dtype, torch.bfloat16)
            self.assertEqual(model.model.visual.seen["interp_indices"].dtype, torch.long)
            self.assertEqual(model.rm_head(torch.zeros(1, 2, dtype=torch.bfloat16)).dtype, torch.float32)
            model.model(image_grid_thw=grid)
            self.assertFalse(model.model.seen["use_cache"])
            model.model(use_cache=True, image_grid_thw=grid)
            self.assertTrue(model.model.seen["use_cache"])
        finally:
            handle.remove()


if __name__ == "__main__":
    unittest.main()
