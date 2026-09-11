import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
from hpsv3_4bit import cli
from hpsv3_4bit.model_source import resolve_model_source
from hpsv3_4bit.hpsv3pp import upstream


class _Cuda:
    def is_available(self): return True
    def synchronize(self): pass
    def reset_peak_memory_stats(self, *_): pass
    def max_memory_allocated(self, *_): return 0


class _Session:
    def __init__(self): self.calls = []
    def caption(self, images):
        self.calls.append(("caption", images.size))
        return "caption-0"
    def score_batch(self, images, prompts, **kwargs):
        self.calls.append(("score_batch", len(images), list(prompts), kwargs))
        return list(range(len(images)))


class CliTests(unittest.TestCase):
    def test_parser_defaults_alias_and_invalid_values(self):
        hps = cli.build_parser("hpsv3").parse_args(["--input", "x", "--output", "y"])
        pp = cli.build_parser("hpsv3pp").parse_args(["--input", "x", "--output", "y", "--merged-dir", "m"])
        self.assertEqual(hps.batch_size, 4)
        self.assertEqual(pp.batch_size, 2)
        self.assertEqual(pp.merged_dir, "m")
        self.assertIsNone(pp.source_dir)
        self.assertFalse(hasattr(hps, "source_dir"))
        for args in (("--batch-size", "0"), ("--iter-step", "1.1")):
            with self.assertRaises(SystemExit):
                cli.build_parser("hpsv3pp").parse_args(["--input", "x", "--output", "y", *args])

    def test_hpsv3pp_flags_and_batched_output(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            paths = []
            for i in range(3):
                image = root / f"{i}.png"
                Image.new("RGB", (2, 2)).save(image)
                paths.append(str(image))
            records = root / "records.json"
            records.write_text(json.dumps([{"id": str(i), "image": p, "prompt": f"p{i}"} for i, p in enumerate(paths)]))
            output = root / "out.json"
            session = _Session()
            fake_torch = types.SimpleNamespace(cuda=_Cuda(), zeros=lambda *a, **k: None)
            with patch.dict(sys.modules, {"torch": fake_torch}), patch.object(upstream, "ensure_source") as ensure, patch.object(cli, "load_model", return_value=session) as load, patch.object(cli, "resolve_model_source", return_value=(root, root)):
                cli.main_hpsv3pp(["--input", str(records), "--output", str(output), "--model", "repo", "--revision", "r1", "--local-files-only", "--source-dir", str(root / "source"), "--processor-dir", str(root), "--batch-size", "2", "--iter-step", "0.25"])
            ensure.assert_called_once_with(local_files_only=True, source_directory=str(root / "source"))
            self.assertEqual(load.call_args.kwargs["source_directory"], str(root / "source"))
            result = json.loads(output.read_text())
            self.assertEqual([row["id"] for row in result["scores"]], ["0", "1", "2"])
            self.assertEqual(session.calls[0][1:], (2, ["p0", "p1"], {"iter_step": 0.25}))
            self.assertEqual(session.calls[1][1:], (1, ["p2"], {"iter_step": 0.25}))

    def test_huggingface_offline_constant_is_forwarded(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            image = root / "image.png"
            Image.new("RGB", (2, 2)).save(image)
            records = root / "records.json"
            records.write_text(json.dumps([{"id": "image", "image": str(image), "prompt": "p"}]))
            session = _Session()
            fake_torch = types.SimpleNamespace(cuda=_Cuda(), zeros=lambda *a, **k: None)
            with patch.dict(sys.modules, {"torch": fake_torch}), \
                 patch.object(cli, "HF_HUB_OFFLINE", True), \
                 patch.object(upstream, "ensure_source") as ensure, \
                 patch.object(cli, "resolve_model_source", return_value=(root, root)) as resolve, \
                 patch.object(cli, "load_model", return_value=session):
                cli.main_hpsv3pp(["--input", str(records), "--output", str(root / "out.json"), "--source-dir", str(root / "source")])
            ensure.assert_called_once_with(local_files_only=True, source_directory=str(root / "source"))
            self.assertTrue(resolve.call_args.kwargs["local_files_only"])

    def test_no_prompt_captions_are_saved(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            image = root / "x.png"
            Image.new("RGB", (2, 2)).save(image)
            records = root / "records.json"
            records.write_text(json.dumps([{"id": "x", "image": str(image)}]))
            output = root / "out.json"
            session = _Session()
            fake_torch = types.SimpleNamespace(cuda=_Cuda(), zeros=lambda *a, **k: None)
            with patch.dict(sys.modules, {"torch": fake_torch}), patch.object(cli, "load_model", return_value=session), patch.object(cli, "resolve_model_source", return_value=(root, root)):
                cli.main_hpsv3(["--input", str(records), "--output", str(output), "--no-prompt"])
            self.assertEqual(json.loads(output.read_text())["scores"][0]["generated_prompt"], "caption-0")

    def test_hub_and_processor_resolution_preserves_revision_and_offline(self):
        with tempfile.TemporaryDirectory() as name:
            model = Path(name) / "model"
            model.mkdir()
            processor = Path(name) / "processor"
            processor.mkdir()
            calls = []
            def download(repo, **kwargs):
                calls.append((repo, kwargs))
                return str(model if len(calls) == 1 else processor)
            with patch("huggingface_hub.snapshot_download", side_effect=download):
                found_model, found_processor = resolve_model_source("hpsv3", "org/model", revision="rev", local_files_only=True)
            self.assertEqual(found_model, model.resolve())
            self.assertEqual(found_processor, processor.resolve())
            self.assertEqual(calls[0][0], "org/model")
            self.assertEqual(calls[0][1]["revision"], "rev")
            self.assertTrue(calls[0][1]["local_files_only"])
            self.assertIn("allow_patterns", calls[1][1])
            self.assertNotIn("*.safetensors", calls[1][1]["allow_patterns"])

    def test_processor_repo_revision_and_independent_repo(self):
        with tempfile.TemporaryDirectory() as name:
            model = Path(name) / "model"
            processor = Path(name) / "processor"
            model.mkdir(); processor.mkdir()
            calls = []
            def download(repo, **kwargs):
                calls.append((repo, kwargs)); return str(model if len(calls) % 2 else processor)
            with patch("huggingface_hub.snapshot_download", side_effect=download):
                resolve_model_source("hpsv3", "org/model", revision="r", processor_dir="org/model")
                resolve_model_source("hpsv3", "org/model", revision="r", processor_dir="org/processor")
            self.assertEqual(calls[1][1]["revision"], "r")
            self.assertIsNone(calls[3][1]["revision"])

    def test_local_and_missing_processor_paths(self):
        with tempfile.TemporaryDirectory() as name:
            model = Path(name) / "model"; processor = Path(name) / "processor"
            model.mkdir(); processor.mkdir()
            found_model, found_processor = resolve_model_source("hpsv3", model, processor_dir=processor)
            self.assertEqual((found_model, found_processor), (model.resolve(), processor.resolve()))
            with self.assertRaises(FileNotFoundError):
                resolve_model_source("hpsv3", model, processor_dir=model / "missing")

    def test_directory_prompt_extension_and_missing_prompt(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name); Image.new("RGB", (2, 2)).save(root / "a.png")
            (root / "a.caption").write_text("hello")
            self.assertEqual(cli.load_records(str(root), prompt_ext=".caption")[0]["prompt"], "hello")
            (root / "a.caption").unlink()
            with self.assertRaises(SystemExit):
                cli.load_records(str(root))

    def test_empty_outputs_keep_family_schema_without_cuda_or_model_load(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name); records = root / "records.json"; records.write_text("[]")
            for family, entry in (("hpsv3", cli.main_hpsv3), ("hpsv3pp", cli.main_hpsv3pp)):
                output = root / f"{family}.json"
                with patch.object(cli, "resolve_model_source", side_effect=AssertionError), patch.object(cli, "load_model", side_effect=AssertionError):
                    entry(["--input", str(records), "--output", str(output)])
                data = json.loads(output.read_text())
                if family == "hpsv3":
                    self.assertEqual(set(data), {"scores", "load_time_sec", "infer_time_sec", "peak_vram_gb"})
                else:
                    self.assertEqual(set(data), {"scores", "load_time_sec", "load_peak_vram_gb", "infer_time_sec", "infer_peak_vram_gb"})

    def test_no_cuda_prevents_model_resolution(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name); image = root / "a.png"; Image.new("RGB", (2, 2)).save(image)
            records = root / "records.json"; records.write_text(json.dumps([{"id": "a", "image": str(image), "prompt": "p"}]))
            fake_torch = types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: False))
            with patch.dict(sys.modules, {"torch": fake_torch}), patch.object(cli, "resolve_model_source", side_effect=AssertionError):
                with self.assertRaises(AssertionError):
                    cli.main_hpsv3(["--input", str(records), "--output", str(root / "out.json")])


if __name__ == "__main__":
    unittest.main()
