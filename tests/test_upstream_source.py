import hashlib
import io
import importlib.util
import os
import py_compile
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from hpsv3_4bit.hpsv3pp import upstream


class _Response:
    def __init__(self, data):
        self.data = io.BytesIO(data)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self, size=-1):
        return self.data.read(size)


class UpstreamSourceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.base = Path(self.tmp.name)
        self.files = {
            "hpsv3/model/qwen3vl_rm.py": b"VALUE = 7\n",
            "hpsv3/dataset/data_collator_qwen.py": (
                b"INSTRUCTION = 'instruction {text_prompt}'\n"
                b"prompt_with_special_token = 'reward'\n"
                b"prompt_without_special_token = 'plain'\n"
            ),
        }
        self.spec = {name: (hashlib.sha256(data).hexdigest(), 1024) for name, data in self.files.items()}

    def tearDown(self):
        self.tmp.cleanup()

    def _patch_source(self):
        return patch.multiple(
            upstream,
            _FILES=self.spec,
            _TIMEOUT=1,
        )

    def test_import_has_no_network_and_local_source_is_loaded(self):
        with self._patch_source():
            target = self.base / upstream.COMMIT
            for relative, data in self.files.items():
                path = target / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(data)
            model_file = target / "hpsv3/model/qwen3vl_rm.py"
            # A valid unchecked-hash pyc would normally override the reviewed
            # source. The external loader must compile only validated bytes.
            model_file.write_bytes(b"VALUE = 9\n")
            py_compile.compile(str(model_file), invalidation_mode=py_compile.PycInvalidationMode.UNCHECKED_HASH)
            model_file.write_bytes(self.files["hpsv3/model/qwen3vl_rm.py"])
            normal_spec = importlib.util.spec_from_file_location("poison_probe", model_file)
            normal_module = importlib.util.module_from_spec(normal_spec)
            normal_spec.loader.exec_module(normal_module)
            self.assertEqual(normal_module.VALUE, 9)
            with patch.object(upstream, "urlopen", side_effect=AssertionError("network")):
                self.assertEqual(upstream.load_prompts(self.base), {"INSTRUCTION": "instruction {text_prompt}", "prompt_with_special_token": "reward", "prompt_without_special_token": "plain"})
                self.assertEqual(upstream.load_model_module(self.base).VALUE, 7)

    def test_download_is_atomic_and_failed_download_leaves_no_partial_source(self):
        with self._patch_source():
            response = _Response(self.files[next(iter(self.files))])
            with patch.object(upstream, "urlopen", return_value=response):
                with self.assertRaises(upstream.SourceProvisionError):
                    upstream.ensure_source(source_directory=self.base)
            self.assertFalse((self.base / upstream.COMMIT).exists())
            self.assertEqual(list(self.base.glob(".*staging-*")), [])

    def test_corrupt_cache_is_repaired_atomically(self):
        with self._patch_source():
            responses = [_Response(self.files[name]) for name in self.files]
            with patch.object(upstream, "urlopen", side_effect=responses):
                target = upstream.ensure_source(source_directory=self.base)
            (target / "hpsv3/model/qwen3vl_rm.py").write_bytes(b"corrupt")
            responses = [_Response(self.files[name]) for name in self.files]
            with patch.object(upstream, "urlopen", side_effect=responses):
                repaired = upstream.ensure_source(source_directory=self.base)
            self.assertEqual(repaired, target)
            self.assertEqual((target / "hpsv3/model/qwen3vl_rm.py").read_bytes(), self.files["hpsv3/model/qwen3vl_rm.py"])
            self.assertEqual(list(self.base.glob(".*invalid-*")), [])

    def test_invalid_hash_is_rejected(self):
        with self._patch_source():
            response = _Response(self.files["hpsv3/model/qwen3vl_rm.py"] + b"bad")
            with patch.object(upstream, "urlopen", return_value=response):
                with self.assertRaises(upstream.SourceProvisionError):
                    upstream.ensure_source(source_directory=self.base)
            self.assertFalse((self.base / upstream.COMMIT).exists())

    def test_interruption_removes_staging(self):
        with self._patch_source():
            with patch.object(upstream, "urlopen", side_effect=KeyboardInterrupt):
                with self.assertRaises(KeyboardInterrupt):
                    upstream.ensure_source(source_directory=self.base)
            self.assertEqual(list(self.base.iterdir()), [])

    def test_failed_repair_publication_restores_old_cache(self):
        with self._patch_source():
            with patch.object(upstream, "urlopen", side_effect=[_Response(data) for data in self.files.values()]):
                target = upstream.ensure_source(source_directory=self.base)
            corrupt = target / "hpsv3/model/qwen3vl_rm.py"
            corrupt.write_bytes(b"original broken cache")
            replace = os.replace
            def fail_publication(source, destination):
                if ".staging-" in Path(source).name:
                    raise OSError("publication failed")
                replace(source, destination)
            with patch.object(upstream, "urlopen", side_effect=[_Response(data) for data in self.files.values()]), \
                 patch.object(upstream.os, "replace", side_effect=fail_publication):
                with self.assertRaisesRegex(OSError, "publication failed"):
                    upstream.ensure_source(source_directory=self.base)
            self.assertEqual(corrupt.read_bytes(), b"original broken cache")
            self.assertEqual(list(self.base.glob(".*")), [])

    def test_linked_cache_is_rejected_without_modifying_destination(self):
        with tempfile.TemporaryDirectory() as outside:
            target = self.base / upstream.COMMIT
            try:
                target.symlink_to(outside, target_is_directory=True)
            except OSError as error:
                self.skipTest(f"Symlinks unavailable: {error}")
            with self._patch_source(), \
                 patch.object(upstream, "urlopen", side_effect=AssertionError("network")):
                with self.assertRaisesRegex(upstream.SourceProvisionError, "symlink"):
                    upstream.ensure_source(source_directory=self.base)
            self.assertEqual(list(Path(outside).iterdir()), [])

    def test_offline_mode_requires_existing_validated_source(self):
        with self._patch_source(), patch.object(upstream, "urlopen", side_effect=AssertionError("network")):
            with self.assertRaises(upstream.SourceProvisionError):
                upstream.ensure_source(local_files_only=True, source_directory=self.base)


if __name__ == "__main__":
    unittest.main()
