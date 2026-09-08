import json
from pathlib import Path
import sys
import tempfile
import types
import unittest

import torch
from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from evaluation._merged_config import load_merged_config


class MergedConfigTests(unittest.TestCase):
    def test_nested_config_is_flattened_and_vocab_follows_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            raw = dict(model_type="qwen2_vl", hidden_size=8, num_hidden_layers=2,
                       num_attention_heads=2, intermediate_size=16, vocab_size=100,
                       vision_config={"depth": 1})
            raw["text_config"] = {k: raw[k] for k in ("hidden_size", "num_hidden_layers", "num_attention_heads", "intermediate_size")}
            (path / "config.json").write_text(json.dumps(raw))
            save_file({"model.embed_tokens.weight": torch.zeros(10, 8),
                       "lm_head.weight": torch.zeros(10, 8)}, str(path / "model.safetensors"))
            original = (path / "config.json").read_bytes()
            tokenizer = types.SimpleNamespace(get_vocab=lambda: {"reward": 9})
            config = load_merged_config(directory, tokenizer)
            self.assertEqual(config.vocab_size, 10)
            self.assertFalse(hasattr(config, "text_config"))
            self.assertEqual((path / "config.json").read_bytes(), original)
            tokenizer.get_vocab = lambda: {"reward": 10}
            with self.assertRaisesRegex(ValueError, "exceeds"):
                load_merged_config(directory, tokenizer)
            raw["text_config"]["hidden_size"] = 16
            (path / "config.json").write_text(json.dumps(raw))
            with self.assertRaisesRegex(ValueError, "Conflicting"):
                load_merged_config(directory, tokenizer)


if __name__ == "__main__":
    unittest.main()
