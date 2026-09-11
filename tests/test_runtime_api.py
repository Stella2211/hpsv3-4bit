import sys
import types
import unittest
import ast
import json
import hashlib
import tempfile
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

from PIL import Image

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
from hpsv3_4bit.runtime import HPSv3Session, load_model


def tiny_config(family):
    from transformers import Qwen2VLConfig, Qwen3VLConfig
    cls = Qwen2VLConfig if family == "hpsv3" else Qwen3VLConfig
    config = cls(
        text_config={"vocab_size": 64, "hidden_size": 32, "intermediate_size": 64,
            "num_hidden_layers": 1, "num_attention_heads": 4, "num_key_value_heads": 4,
            "rope_parameters": {"rope_type": "default", "mrope_section": [2, 1, 1]}},
        vision_config={"depth": 1, "embed_dim": 32, "hidden_size": 32, "out_hidden_size": 32,
            "intermediate_size": 64, "num_heads": 4, "patch_size": 14,
            "spatial_merge_size": 2, "temporal_patch_size": 2},
        image_token_id=10, video_token_id=11, vision_start_token_id=12, vision_end_token_id=13,
    )
    config.pad_token_id = 0
    return config


class FakeInferencer:
    def __init__(self):
        self.model = object()
        self.calls = []

    def score(self, images, prompts, **kwargs):
        self.calls.append(("score", images, prompts, kwargs))
        return [1.25] * len(images)

    def caption(self, images, **kwargs):
        self.calls.append(("caption", images, kwargs))
        return ["a test caption"]


class RuntimeApiTests(unittest.TestCase):
    def setUp(self):
        import hpsv3_4bit.hpsv3pp.quantized as pp_quantized
        self._prompt_patch = patch.object(
            pp_quantized,
            "load_prompts",
            return_value={"INSTRUCTION": "{text_prompt}", "prompt_with_special_token": "<|Reward|>"},
        )
        self._prompt_patch.start()
        self.addCleanup(self._prompt_patch.stop)

    def test_session_is_single_pair_and_checks_cancellation(self):
        checks = []
        inferencer = FakeInferencer()
        session = HPSv3Session("hpsv3pp", inferencer, lambda: checks.append(True))
        image = Image.new("RGB", (28, 28))

        self.assertEqual(session.score(image, "prompt"), 1.25)
        self.assertEqual(session.caption(image), "a test caption")
        self.assertEqual(len(checks), 4)
        self.assertEqual(inferencer.calls[0][3], {"iter_step": 0.0})

    def test_session_preserves_batch_order_and_iteration(self):
        inferencer = FakeInferencer()
        session = HPSv3Session("hpsv3pp", inferencer)
        images = [Image.new("RGB", (28, 28), color) for color in ("red", "blue")]
        self.assertEqual(session.score_batch(images, ["red", "blue"], iter_step=0.5), [1.25, 1.25])
        self.assertEqual(inferencer.calls, [("score", images, ["red", "blue"], {"iter_step": 0.5})])

    def test_session_rejects_invalid_batch_and_results(self):
        from unittest.mock import Mock
        image = Image.new("RGB", (28, 28))
        inferencer = FakeInferencer()
        session = HPSv3Session("hpsv3pp", inferencer)
        for images, prompts, iteration in (([], [], 0), ([image], [], 0),
                                            ([image], ["a"], float("nan")), ([image], ["a"], 2)):
            with self.assertRaises(ValueError):
                session.score_batch(images, prompts, iter_step=iteration)
        self.assertEqual(inferencer.calls, [])
        for result in ([], [float("nan")], [float("inf")]):
            inferencer.score = Mock(return_value=result)
            with self.assertRaises((ValueError, RuntimeError)):
                session.score(image, "a")
        with self.assertRaisesRegex(ValueError, "does not support"):
            HPSv3Session("hpsv3", FakeInferencer()).score_batch([image], ["a"], iter_step=0.5)

    def test_load_model_rejects_unknown_family(self):
        with self.assertRaisesRegex(ValueError, "Unknown HPS model family"):
            load_model("other", ".")

    def test_load_model_uses_local_only_family_loader(self):
        fake = FakeInferencer()
        module = types.ModuleType("hpsv3_4bit.hpsv3.quantized")
        loader = types.SimpleNamespace(from_merged_dir=lambda **kwargs: fake)
        module.HPSv3QuantizedInferencer = loader
        with patch.dict(sys.modules, {"hpsv3_4bit.hpsv3.quantized": module}):
            session = load_model("hpsv3", "/tmp/model", "cpu")
        self.assertIs(session.inferencer, fake)

    def test_missing_local_model_is_rejected_without_download(self):
        from hpsv3_4bit.hpsv3.quantized import HPSv3QuantizedInferencer
        with self.assertRaises(FileNotFoundError):
            HPSv3QuantizedInferencer.from_merged_dir(Path("does-not-exist"))

    def test_external_processor_reward_token_is_added_before_validation(self):
        import importlib
        from unittest.mock import Mock
        for family, name in (("hpsv3", "HPSv3QuantizedInferencer"), ("hpsv3pp", "HPSv3PPQuantizedInferencer")):
            module = importlib.import_module(f"hpsv3_4bit.{family}.quantized")
            tokenizer = Mock()
            tokenizer.get_vocab.return_value = {"base": 1}
            processor = types.SimpleNamespace(tokenizer=tokenizer)
            def validate(*args):
                tokenizer.add_special_tokens.assert_called_once_with({"additional_special_tokens": ["<|Reward|>"]})
                raise ValueError("validation reached")
            with tempfile.TemporaryDirectory() as model_dir, tempfile.TemporaryDirectory() as processor_dir:
                with patch.object(module, "_load_processor", return_value=processor) as loader, \
                     patch.object(module, "_validate_checkpoint", side_effect=validate):
                    with self.assertRaisesRegex(ValueError, "validation reached"):
                        getattr(module, name).from_merged_dir(model_dir, processor_directory=processor_dir)
                    loader.assert_called_once_with(Path(processor_dir).resolve())

    def test_visual_dtype_uses_floating_patch_embedding_not_packed_weights(self):
        import torch
        from hpsv3_4bit.hpsv3.quantized import _patch_quantized_visual_dtype
        visual = types.SimpleNamespace(patch_embed=types.SimpleNamespace(
            proj=types.SimpleNamespace(weight=torch.ones(1, dtype=torch.bfloat16))),
            get_dtype=lambda: torch.uint8)
        model = types.SimpleNamespace(model=types.SimpleNamespace(visual=visual))
        _patch_quantized_visual_dtype(model)
        self.assertEqual(visual.get_dtype(), torch.bfloat16)
        pixels = torch.tensor([-1.5, 0.5, 1.5])
        self.assertTrue(torch.equal(pixels.to(visual.get_dtype()).float(), pixels))

    def test_hpsv3_loader_passes_strict_local_flags(self):
        import hpsv3_4bit.hpsv3.quantized as module
        class Tokenizer:
            pad_token_id = 0
            def convert_tokens_to_ids(self, token): return 7
        class Processor:
            tokenizer = Tokenizer()
        class Model:
            rm_head = types.SimpleNamespace(float=lambda: None)
            def eval(self): return self
        captured = {}
        with tempfile.TemporaryDirectory() as name:
            directory = Path(name)
            (directory / "config.json").write_text(json.dumps({"model_type": "qwen2_vl", "quantization_config": {
                "quant_method": "bitsandbytes", "bnb_4bit_quant_type": "nf4", "load_in_4bit": True}}))
            settings = {"output_dim": 2, "reward_token": "special", "special_token_ids": [7],
                        "rm_head_type": "ranknet", "rm_head_kwargs": None}
            (directory / "reward_config.json").write_text(json.dumps({"format_version": 1,
                "reward_token": "<|Reward|>", "reward_token_id": 7, "model_kwargs": settings}))
            def load(*args, **kwargs):
                captured.update(kwargs)
                return Model(), {"missing_keys": [], "mismatched_keys": [], "unexpected_keys": [], "error_msgs": []}
            with patch.object(module.AutoProcessor, "from_pretrained", return_value=Processor()), \
                 patch.object(module.Qwen2VLRewardModelBT, "from_pretrained", side_effect=load), \
                 patch.object(module, "load_merged_config", return_value=types.SimpleNamespace()), \
                 patch.object(module, "_patch_quantized_visual_dtype"):
                module.HPSv3QuantizedInferencer.from_merged_dir(directory, device="cpu")
        self.assertTrue(captured["local_files_only"])
        self.assertFalse(captured["trust_remote_code"])
        self.assertTrue(captured["use_safetensors"])
        self.assertTrue(captured["output_loading_info"])
        self.assertEqual(captured["config"].pad_token_id, 0)
        self.assertFalse(captured["config"].use_cache)

    def test_invalid_nf4_and_reward_settings_are_rejected(self):
        import hpsv3_4bit.hpsv3.quantized as module
        class Tokenizer:
            def convert_tokens_to_ids(self, token): return 7
        processor = types.SimpleNamespace(tokenizer=Tokenizer())
        with tempfile.TemporaryDirectory() as name:
            directory = Path(name)
            (directory / "config.json").write_text(json.dumps({"model_type": "qwen2_vl", "quantization_config": {}}))
            (directory / "reward_config.json").write_text(json.dumps({"format_version": 1, "model_kwargs": {}}))
            with self.assertRaises(ValueError):
                module._validate_checkpoint(directory, processor)

    def test_hpsv3pp_loader_passes_strict_local_flags(self):
        import hpsv3_4bit.hpsv3pp.quantized as module
        class Tokenizer:
            pad_token_id = 0
            def convert_tokens_to_ids(self, token): return 7
        class Processor:
            tokenizer = Tokenizer()
        class Model:
            rm_head = types.SimpleNamespace(float=lambda: None)
            def eval(self): return self
        captured = {}
        with tempfile.TemporaryDirectory() as name:
            directory = Path(name)
            (directory / "config.json").write_text(json.dumps({"model_type": "qwen3_vl", "quantization_config": {
                "quant_method": "bitsandbytes", "bnb_4bit_quant_type": "nf4", "load_in_4bit": True}}))
            settings = {"output_dim": 2, "reward_token": "special", "special_token_ids": [7],
                        "rm_head_type": "ranknet", "rm_head_kwargs": None, "cond_dim": 256}
            (directory / "reward_config.json").write_text(json.dumps({"format_version": 1,
                "reward_token": "<|Reward|>", "reward_token_id": 7, "model_kwargs": settings}))
            def load(*args, **kwargs):
                captured.update(kwargs)
                return Model(), {"missing_keys": [], "mismatched_keys": [], "unexpected_keys": [], "error_msgs": []}
            fake_class = types.SimpleNamespace(from_pretrained=staticmethod(load))
            with patch.object(module.AutoProcessor, "from_pretrained", return_value=Processor()), \
                 patch.object(module, "get_reward_model_class", return_value=fake_class), \
                 patch.object(module, "install_vision_interpolation_hook"), \
                 patch.object(module, "_restore_capability_dtype"), \
                 patch.object(module, "load_merged_config", return_value=types.SimpleNamespace()):
                module.HPSv3PPQuantizedInferencer.from_merged_dir(directory, device="cpu")
        self.assertTrue(captured["local_files_only"])
        self.assertFalse(captured["trust_remote_code"])
        self.assertTrue(captured["use_safetensors"])
        self.assertTrue(captured["output_loading_info"])
        self.assertEqual(captured["config"].pad_token_id, 0)
        self.assertFalse(captured["config"].use_cache)

    def test_quantized_prepare_batch_validates_each_row_and_pairing(self):
        import importlib
        import torch
        processor = types.SimpleNamespace(tokenizer=types.SimpleNamespace(convert_tokens_to_ids=lambda token: 7))
        images = [Image.new("RGB", (28, 28)), Image.new("RGB", (28, 28))]
        for family, class_name in (("hpsv3", "HPSv3QuantizedInferencer"), ("hpsv3pp", "HPSv3PPQuantizedInferencer")):
            module = importlib.import_module(f"hpsv3_4bit.{family}.quantized")
            inferencer = getattr(module, class_name)(object(), processor, "cpu")
            for pictures, prompts in (([], []), (images, ["a"])):
                with self.assertRaisesRegex(ValueError, "one prompt per image"):
                    inferencer.prepare_batch(pictures, prompts)
            good = {"input_ids": torch.tensor([[1, 7, 2], [7, 2, 0]])}
            with patch.object(module, "_batch", return_value=good) as batch:
                self.assertIs(inferencer.prepare_batch(images, ["a", "b"]), good)
                self.assertEqual(batch.call_args.args[1], images)
            # Total reward token count alone cannot catch a missing token in
            # one row combined with two tokens in another.
            with patch.object(module, "_batch", return_value={"input_ids": torch.tensor([[7, 7], [1, 2]])}):
                with self.assertRaisesRegex(ValueError, "exactly one reward token"):
                    inferencer.prepare_batch(images, ["a", "b"])

    def test_incomplete_checkpoint_loads_are_never_accepted(self):
        import importlib
        for family, class_name in (("hpsv3", "HPSv3QuantizedInferencer"), ("hpsv3pp", "HPSv3PPQuantizedInferencer")):
            module = importlib.import_module(f"hpsv3_4bit.{family}.quantized")
            if family == "hpsv3":
                model_class = module.Qwen2VLRewardModelBT
            else:
                model_class = types.SimpleNamespace(from_pretrained=staticmethod(lambda *args, **kwargs: (object(), {})))
            for field in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs"):
                with self.subTest(family=family, field=field), tempfile.TemporaryDirectory() as directory:
                    patches = [
                        patch.object(module, "_load_processor", return_value=types.SimpleNamespace(tokenizer=types.SimpleNamespace(pad_token_id=0))),
                        patch.object(module, "_validate_checkpoint", return_value={}),
                        patch.object(module, "load_merged_config", return_value=types.SimpleNamespace()),
                        patch.object(model_class, "from_pretrained", return_value=(object(), {field: ["bad weight"]})),
                    ]
                    if family == "hpsv3pp":
                        patches.append(patch.object(module, "get_reward_model_class", return_value=model_class))
                    with ExitStack() as stack:
                        for item in patches:
                            stack.enter_context(item)
                        with self.assertRaisesRegex(ValueError, "checkpoint mismatch"):
                            getattr(module, class_name).from_merged_dir(directory, device="cpu")

    def test_capability_encoder_retains_saved_bf16_precision(self):
        import torch
        from safetensors.torch import save_file
        from hpsv3_4bit.hpsv3pp.quantized import _restore_capability_dtype
        model = types.SimpleNamespace(cap_encoder=torch.nn.Linear(2, 1))
        with tempfile.TemporaryDirectory() as directory:
            save_file({"cap_encoder.proj.weight": torch.ones(1, 2, dtype=torch.bfloat16)}, str(Path(directory) / "model.safetensors"))
            _restore_capability_dtype(model, Path(directory))
        self.assertEqual(model.cap_encoder.weight.dtype, torch.bfloat16)

    def test_prompt_cannot_add_a_second_reward_token(self):
        import importlib
        import torch
        processor = types.SimpleNamespace(tokenizer=types.SimpleNamespace(convert_tokens_to_ids=lambda token: 7))
        for family, class_name in (("hpsv3", "HPSv3QuantizedInferencer"), ("hpsv3pp", "HPSv3PPQuantizedInferencer")):
            module = importlib.import_module(f"hpsv3_4bit.{family}.quantized")
            inferencer = getattr(module, class_name)(object(), processor, "cpu")
            with patch.object(module, "_batch", return_value={"input_ids": torch.tensor([[7, 2, 7]])}):
                with self.assertRaisesRegex(ValueError, "exactly one reward token"):
                    inferencer.prepare_batch([Image.new("RGB", (28, 28))], ["<|Reward|>"])

    def test_caption_restores_forward_after_generation_error(self):
        import torch
        import hpsv3_4bit.hpsv3.quantized as module
        class Model:
            def generate(self, **kwargs): raise RuntimeError("stop")
        processor = types.SimpleNamespace(tokenizer=types.SimpleNamespace(pad_token_id=0, decode=lambda *a, **k: ""))
        inferencer = module.HPSv3QuantizedInferencer(Model(), processor, "cpu")
        with patch.object(module, "_batch", return_value={"input_ids": torch.ones(1, 1, dtype=torch.long)}):
            with self.assertRaisesRegex(RuntimeError, "stop"):
                inferencer.caption([Image.new("RGB", (28, 28))])
        self.assertNotIn("forward", vars(inferencer.model))

    def test_hpsv3pp_prompt_constants_match_pinned_upstream(self):
        package = Path(__file__).parents[1]
        from hpsv3_4bit.hpsv3pp.prompts import load_prompts
        from hpsv3_4bit.hpsv3pp.upstream import SourceProvisionError
        try:
            prompts = load_prompts()
        except (FileNotFoundError, ImportError, SourceProvisionError) as exc:
            self.skipTest(f"external upstream source unavailable: {exc}")
        self.assertIn("INSTRUCTION", prompts)
        self.assertIn("prompt_with_special_token", prompts)
        self.assertEqual(hashlib.sha256(prompts["INSTRUCTION"].encode()).hexdigest(),
                         "4b760179e6eeec994804cc37e30968d0132bedeea81989ca7b045f8a2051397d")
        self.assertEqual(hashlib.sha256(prompts["prompt_with_special_token"].encode()).hexdigest(),
                         "2471faca094ae5bd6fe17b71dce20c3d7583625545e235889f7e74e7eeb69d0a")

    def test_hpsv3_prompt_constants_match_legacy_protocol(self):
        package = Path(__file__).parents[1]
        canonical = package / "src" / "hpsv3_4bit" / "hpsv3" / "quantized.py"
        def constants(path):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            found = {}
            for node in tree.body:
                if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                    if node.targets[0].id in {"INSTRUCTION", "PROMPT_WITH_SPECIAL_TOKEN"}:
                        found[node.targets[0].id] = hashlib.sha256(ast.literal_eval(node.value).encode()).hexdigest()
            return found
        # Protocol from the pre-migration 43bdde3 implementation, including
        # exact whitespace; retain it without keeping a second inferencer.
        self.assertEqual(constants(canonical), {
            "INSTRUCTION": "32a5082d2b86dcc19a98aa5b24890ff2a666ada52c0bfc4b0516681f4db3e478",
            "PROMPT_WITH_SPECIAL_TOKEN": "2471faca094ae5bd6fe17b71dce20c3d7583625545e235889f7e74e7eeb69d0a",
        })

    def test_hpsv3_reward_model_forward_uses_public_backbone(self):
        import torch
        from hpsv3_4bit.hpsv3.model import Qwen2VLRewardModelBT
        model = Qwen2VLRewardModelBT(tiny_config("hpsv3"), output_dim=2, reward_token="special",
                                     special_token_ids=[7], rm_head_type="ranknet")
        model.eval()
        result = model(input_ids=torch.tensor([[1, 7, 2]]), attention_mask=torch.ones(1, 3, dtype=torch.long),
                       mm_token_type_ids=torch.zeros(1, 3, dtype=torch.long))
        self.assertEqual(tuple(result["logits"].shape), (1, 2))

    def test_hpsv3pp_reward_model_forward_uses_public_backbone(self):
        import torch
        from transformers import Qwen3VLConfig
        from hpsv3_4bit.hpsv3pp.upstream import SourceProvisionError
        try:
            from hpsv3_4bit.hpsv3pp.model import Qwen3VLRewardModelFiLMHybrid
        except (FileNotFoundError, ImportError, SourceProvisionError) as exc:
            self.skipTest(f"external upstream source unavailable: {exc}")
        config = Qwen3VLConfig(
            text_config={"vocab_size": 32, "hidden_size": 8, "intermediate_size": 16,
                "num_hidden_layers": 1, "num_attention_heads": 2, "num_key_value_heads": 2,
                "rope_scaling": {"rope_type": "default", "mrope_section": [2, 1, 1]}},
            vision_config={"depth": 1, "hidden_size": 8, "intermediate_size": 16,
                "num_heads": 2, "patch_size": 14, "spatial_merge_size": 2, "temporal_patch_size": 2},
            image_token_id=10, video_token_id=11, vision_start_token_id=12, vision_end_token_id=13)
        config.pad_token_id = 0
        model = Qwen3VLRewardModelFiLMHybrid(config, output_dim=2, reward_token="last",
                                              rm_head_type="ranknet", cond_dim=4)
        result = model(input_ids=torch.tensor([[1, 2, 3, 0]]),
                       attention_mask=torch.ones(1, 4, dtype=torch.long),
                       iter_values=torch.zeros(1))
        self.assertEqual(tuple(result["logits"].shape), (1, 2))

    def test_bf16_backbone_loading_preserves_fp32_reward_weights(self):
        import torch
        from hpsv3_4bit.hpsv3.model import Qwen2VLRewardModelBT
        from hpsv3_4bit.hpsv3pp.upstream import SourceProvisionError
        try:
            from hpsv3_4bit.hpsv3pp.model import Qwen3VLRewardModelFiLMHybrid
        except (FileNotFoundError, ImportError, SourceProvisionError) as exc:
            self.skipTest(f"external upstream source unavailable: {exc}")
        for family, cls in (("hpsv3", Qwen2VLRewardModelBT), ("hpsv3pp", Qwen3VLRewardModelFiLMHybrid)):
            with self.subTest(family=family), tempfile.TemporaryDirectory() as directory:
                settings = dict(output_dim=2, reward_token="special", special_token_ids=[7], rm_head_type="ranknet")
                if family == "hpsv3pp":
                    settings["cond_dim"] = 4
                model = cls(tiny_config(family), **settings)
                names = ("rm_head",) if family == "hpsv3" else ("rm_head", "cond_encoder", "film_gen")
                original = {}
                with torch.no_grad():
                    for name in names:
                        parameter = next(getattr(model, name).parameters())
                        parameter.fill_(0.1234567)
                        original[name] = parameter.clone()
                model.save_pretrained(directory)
                loaded = cls.from_pretrained(directory, **settings, dtype=torch.bfloat16,
                    local_files_only=True, trust_remote_code=False, use_safetensors=True)
                for name in names:
                    parameter = next(getattr(loaded, name).parameters())
                    self.assertEqual(parameter.dtype, torch.float32)
                    self.assertTrue(torch.equal(parameter, original[name]))
                self.assertEqual(loaded.get_input_embeddings().weight.dtype, torch.bfloat16)


if __name__ == "__main__":
    unittest.main()
