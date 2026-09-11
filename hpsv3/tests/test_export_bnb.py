"""Regression checks for the conversion-only NF4 export path."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from evaluation import _conversion as conversion


class ExportTests(unittest.TestCase):
    def test_bf16_and_serialized_inputs_export_loadable_reward_metadata(self):
        for saved in (None, {"quant_method": "bitsandbytes"}):
            with self.subTest(saved=saved), tempfile.TemporaryDirectory() as tmp:
                model = Mock()
                processor = Mock()
                processor.tokenizer.get_vocab.return_value = {"<|Reward|>": 7}
                processor.tokenizer.convert_tokens_to_ids.return_value = [7]
                with patch.object(conversion, "resolve_model", return_value="merged"), \
                     patch.object(conversion.AutoProcessor, "from_pretrained", return_value=processor), \
                     patch.object(conversion, "load_reward_settings", side_effect=lambda directory, defaults, tokenizer: defaults), \
                     patch.object(conversion, "saved_quantization_config", return_value=saved), \
                     patch.object(conversion, "load_merged_config", return_value=Mock()), \
                     patch.object(conversion.Qwen2VLRewardModelBT, "from_pretrained", return_value=model) as load:
                    conversion.export_bnb("merged", tmp)
                qconfig = load.call_args.kwargs["quantization_config"]
                if saved is None:
                    self.assertTrue(qconfig.load_in_4bit)
                    self.assertEqual(qconfig.bnb_4bit_quant_type, "nf4")
                    self.assertTrue(qconfig.bnb_4bit_use_double_quant)
                    self.assertEqual(qconfig.bnb_4bit_compute_dtype, conversion.torch.bfloat16)
                    expected_skip = ["rm_head", "lm_head"] + getattr(conversion, "_FP32_COND_ATTRS", [])
                    self.assertTrue(set(expected_skip) <= set(qconfig.llm_int8_skip_modules))
                else:
                    self.assertIsNone(qconfig)
                model.save_pretrained.assert_called_once_with(tmp, safe_serialization=True, max_shard_size="2GB")
                processor.save_pretrained.assert_called_once_with(tmp)
                metadata = json.loads((Path(tmp) / "reward_config.json").read_text())
                self.assertEqual(metadata["format_version"], 1)
                self.assertEqual(metadata["reward_token_id"], 7)
                self.assertEqual(metadata["model_kwargs"]["special_token_ids"], [7])
                self.assertEqual(metadata["model_kwargs"]["output_dim"], 2)

    def test_cli_and_export_reject_nonempty_output_before_loading(self):
        script = Path(__file__).resolve().parents[1] / "scripts" / "export_bnb.py"
        spec = importlib.util.spec_from_file_location("export_cli", script)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "existing.txt").write_text("keep")
            with patch.object(conversion, "resolve_model") as resolve:
                with self.assertRaisesRegex(ValueError, "must be empty"):
                    conversion.export_bnb("unused", tmp)
                with self.assertRaisesRegex(ValueError, "must be empty"):
                    module.main(["--merged-dir", "unused", "--output-dir", tmp])
                resolve.assert_not_called()
            self.assertEqual((Path(tmp) / "existing.txt").read_text(), "keep")

    def test_non_bitsandbytes_serialization_is_rejected(self):
        processor = Mock()
        processor.tokenizer.get_vocab.return_value = {"<|Reward|>": 7}
        with tempfile.TemporaryDirectory() as tmp, \
             patch.object(conversion, "resolve_model", return_value="merged"), \
             patch.object(conversion.AutoProcessor, "from_pretrained", return_value=processor), \
             patch.object(conversion, "saved_quantization_config", return_value={"quant_method": "gptq"}), \
             patch.object(conversion.Qwen2VLRewardModelBT, "from_pretrained") as load:
            with self.assertRaisesRegex(ValueError, "bitsandbytes"):
                conversion.export_bnb("merged", tmp)
            load.assert_not_called()


if __name__ == "__main__":
    unittest.main()
