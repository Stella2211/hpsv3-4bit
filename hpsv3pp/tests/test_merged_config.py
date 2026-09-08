import json
from pathlib import Path
import sys
import tempfile
import unittest

from transformers import Qwen3VLConfig

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from evaluation._merged_config import load_merged_config


class MergedConfigTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name)

    def test_valid_config_preserved(self):
        config = Qwen3VLConfig(text_config={"vocab_size": 123, "rope_scaling": {"rope_type": "default"}})
        config.save_pretrained(self.path)
        original = (self.path / "config.json").read_bytes()
        loaded = load_merged_config(self.path, None)
        self.assertEqual(loaded.to_dict(), config.to_dict())
        self.assertEqual((self.path / "config.json").read_bytes(), original)

    def test_training_config_rejected_without_modifying_files(self):
        raw = dict(model_type="film_hybrid", model_name_or_path="Qwen3-VL-8B-Instruct",
                   cond_dim=256, output_dim=2, rm_head_type="ranknet")
        config_path = self.path / "config.json"
        config_path.write_text(json.dumps(raw))
        original = config_path.read_bytes()
        with self.assertRaisesRegex(ValueError, "Legacy training-only configs are not supported"):
            load_merged_config(self.path, None)
        self.assertEqual(config_path.read_bytes(), original)

    def test_missing_or_wrong_architecture_rejected(self):
        for raw in (
            {"model_type": "other", "text_config": {"vocab_size": 123}, "vision_config": {"depth": 2}},
            {"model_type": "qwen3_vl"},
            {"model_type": "qwen3_vl", "text_config": {}, "vision_config": {}},
            {"model_type": "qwen3_vl", "text_config": [], "vision_config": {}},
        ):
            with self.subTest(raw=raw):
                (self.path / "config.json").write_text(json.dumps(raw))
                with self.assertRaisesRegex(ValueError, "requires a Qwen3-VL config.json"):
                    load_merged_config(self.path, None)


if __name__ == "__main__":
    unittest.main()
