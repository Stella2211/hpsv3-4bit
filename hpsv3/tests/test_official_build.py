import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

import torch
from safetensors.torch import load_file, save_file
from transformers import Qwen2VLConfig

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "hpsv3/src"))
from evaluation.hpsv3_model import Qwen2VLRewardModelBT

spec = importlib.util.spec_from_file_location("official_build", ROOT / "scripts/build_official_bf16.py")
official_build = importlib.util.module_from_spec(spec)
spec.loader.exec_module(official_build)


class OfficialBuildTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name)
        self.config = Qwen2VLConfig(vocab_size=10, hidden_size=16, intermediate_size=32,
            num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2,
            vision_config=dict(depth=1, embed_dim=16, hidden_size=16, num_heads=2,
                               mlp_ratio=2, patch_size=2, spatial_patch_size=2))
        model = Qwen2VLRewardModelBT(self.config, output_dim=2, reward_token="special",
                                   special_token_ids=[9], rm_head_type="ranknet")
        model.to(torch.bfloat16)
        model.rm_head.to(torch.float32)
        self.state = model.state_dict()

        class Tokenizer:
            pad_token_id = 0

            def __len__(self):
                return 10

            def add_special_tokens(self, value):
                return 1

            def convert_tokens_to_ids(self, value):
                return 9

        self.processor = Mock(tokenizer=Tokenizer())

    def run_build(self):
        source = self.path / "source.safetensors"
        save_file(self.state, str(source))
        with patch("transformers.AutoProcessor.from_pretrained", return_value=self.processor), \
                patch("transformers.AutoConfig.from_pretrained", return_value=self.config):
            official_build.build("hpsv3", str(source), "unused", str(self.path / "output"))

    def test_preserves_every_tensor_and_dtype(self):
        self.run_build()
        index = json.loads((self.path / "output/model.safetensors.index.json").read_text())
        actual = {}
        for name in set(index["weight_map"].values()):
            actual.update(load_file(str(self.path / "output" / name)))
        self.assertEqual(set(actual), set(self.state))
        for key, expected in self.state.items():
            self.assertEqual(actual[key].dtype, expected.dtype)
            self.assertTrue(torch.equal(actual[key], expected), key)

    def test_incomplete_checkpoint_rejected_before_writing(self):
        self.state.pop("rm_head.0.weight")
        with self.assertRaisesRegex(ValueError, "Checkpoint keys differ"):
            self.run_build()
        self.assertFalse((self.path / "output").exists())


if __name__ == "__main__":
    unittest.main()
