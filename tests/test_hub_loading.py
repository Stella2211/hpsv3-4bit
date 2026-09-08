"""CPU-only checks shared by the independent model projects."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]


class HubLoadingTests(unittest.TestCase):
    def helpers(self):
        for project in ("hpsv3", "hpsv3pp"):
            spec = importlib.util.spec_from_file_location(project + "_hub", ROOT / project / "src/evaluation/_hub.py")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            yield module

    def test_default_downloads_full_repo_and_revision_is_optional(self):
        for module in self.helpers():
            download = Mock(return_value="snapshot")
            with patch.dict(sys.modules, huggingface_hub=types.SimpleNamespace(snapshot_download=download)):
                self.assertEqual(module.resolve_model(local_files_only=True), "snapshot")
                kwargs = download.call_args.kwargs
                self.assertEqual(download.call_args.args, (module.DEFAULT_MODEL_ID,))
                self.assertIsNone(kwargs["revision"])
                self.assertTrue(kwargs["local_files_only"])
                self.assertNotIn("allow_patterns", kwargs)
                self.assertNotIn("ignore_patterns", kwargs)
                module.resolve_model(revision="release-tag")
                self.assertEqual(download.call_args.kwargs["revision"], "release-tag")
                module.resolve_model("someone/model", revision="abc")
                self.assertEqual(download.call_args.kwargs["revision"], "abc")
                module.resolve_model("someone/model")
                self.assertIsNone(download.call_args.kwargs["revision"])

    def test_local_directory_does_not_contact_hub(self):
        with tempfile.TemporaryDirectory() as directory:
            for module in self.helpers():
                self.assertEqual(module.resolve_model(directory), directory)
                with self.assertRaises(FileNotFoundError):
                    module.resolve_model(str(Path(directory) / "missing"))

    def test_reward_token_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            defaults = {"special_token_ids": [7], "output_dim": 2}
            tokenizer = types.SimpleNamespace(convert_tokens_to_ids=lambda _: 7)
            path = Path(directory) / "reward_config.json"
            for module in self.helpers():
                data = dict(format_version=1, reward_token="<|Reward|>", reward_token_id=8, model_kwargs=defaults)
                path.write_text(json.dumps(data))
                with self.assertRaisesRegex(ValueError, "token ID"):
                    module.load_reward_settings(directory, defaults, tokenizer)
                data["reward_token_id"] = 7
                path.write_text(json.dumps(data))
                self.assertEqual(module.load_reward_settings(directory, defaults, tokenizer), defaults)


if __name__ == "__main__":
    unittest.main()
