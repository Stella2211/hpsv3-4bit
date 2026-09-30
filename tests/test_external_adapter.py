import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
from hpsv3_4bit.hpsv3pp import model as adapter


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
    def test_class_is_thin_adapter(self):
        module = types.SimpleNamespace(Qwen3VLRewardModelFiLMHybrid=_ExternalBase)
        with patch.object(adapter, "load_model_module", return_value=module) as load:
            cls = adapter.get_reward_model_class("/tmp/source")
        load.assert_called_once_with("/tmp/source")
        self.assertTrue(issubclass(cls, _ExternalBase))
        self.assertEqual(cls._keep_in_fp32_modules_strict, ["rm_head", "cond_encoder", "film_gen"])

    def test_visual_hook_casts_interpolation_weights_to_position_dtype(self):
        model = types.SimpleNamespace(model=_Backbone())
        adapter.install_vision_interpolation_hook(model)
        grid = torch.tensor([[1, 2, 2]])
        model.model.visual(torch.zeros(4, 2), grid)
        self.assertEqual(model.model.visual.seen["interp_weights"].dtype, torch.bfloat16)
        self.assertEqual(model.model.visual.seen["interp_indices"].dtype, torch.long)
        model.model(image_grid_thw=grid)
        self.assertFalse(model.model.seen["use_cache"])
        model.model(use_cache=True, image_grid_thw=grid)
        self.assertTrue(model.model.seen["use_cache"])


if __name__ == "__main__":
    unittest.main()
