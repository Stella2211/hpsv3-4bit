"""Package a complete official reward checkpoint with its Qwen processor/config.

Run in the selected scorer's environment. No base-model weights are needed:
every tensor must match the reward model's full state dictionary. Input tensors
retain their saved dtype. No model assets are downloaded by this script.
"""
import argparse
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import sys


def build(project: str, checkpoint: str, base_model: str, output_dir: str,
          expected_sha256: str | None = None) -> None:
    import torch
    from accelerate import init_empty_weights
    from safetensors import safe_open
    from safetensors.torch import save_file
    from transformers import AutoConfig, AutoProcessor

    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Output directory must be empty")
    with open(checkpoint, "rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError("Official checkpoint SHA-256 mismatch")
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / project / "src"))
    if project == "hpsv3pp":
        from evaluation._conversion import Qwen3VLRewardModelFiLMHybrid as Model
        embedding_key = "model.language_model.embed_tokens.weight"
    else:
        from evaluation.hpsv3_model import Qwen2VLRewardModelBT as Model
        embedding_key = "model.embed_tokens.weight"

    processor = AutoProcessor.from_pretrained(base_model, local_files_only=True, padding_side="right")
    processor.tokenizer.add_special_tokens({"additional_special_tokens": ["<|Reward|>"]})
    token_id = processor.tokenizer.convert_tokens_to_ids("<|Reward|>")
    config = AutoConfig.from_pretrained(base_model, local_files_only=True)
    expected_type = "qwen3_vl" if project == "hpsv3pp" else "qwen2_vl"
    if config.model_type != expected_type:
        raise ValueError(f"Expected {expected_type} architecture")
    settings = dict(output_dim=2, reward_token="special", special_token_ids=[token_id],
                    rm_head_type="ranknet", rm_head_kwargs=None)
    if project == "hpsv3pp":
        settings["cond_dim"] = 256

    with ExitStack() as stack:
        if checkpoint.endswith(".safetensors"):
            source = stack.enter_context(safe_open(checkpoint, framework="pt", device="cpu"))
            keys = set(source.keys())
            tensor = source.get_tensor
            shape = lambda key: tuple(source.get_slice(key).get_shape())
        else:
            source = torch.load(checkpoint, map_location="cpu", mmap=True, weights_only=True)
            if "model" in source:
                source = source["model"]
            keys = set(source)
            tensor = source.__getitem__
            shape = lambda key: tuple(source[key].shape)

        embedding = shape(embedding_key)
        if embedding != shape("lm_head.weight") or embedding[0] != len(processor.tokenizer):
            raise ValueError("Checkpoint vocabulary does not match the official tokenizer plus reward token")
        text_config = config.text_config if project == "hpsv3pp" else config
        text_config.vocab_size = embedding[0]
        config.pad_token_id = processor.tokenizer.pad_token_id
        config.use_cache = False
        with init_empty_weights():
            model = Model(config, **settings)
        expected = model.state_dict()
        if keys != set(expected):
            raise ValueError(f"Checkpoint keys differ: missing={sorted(set(expected)-keys)}, extra={sorted(keys-set(expected))}")
        for key, value in expected.items():
            if shape(key) != tuple(value.shape):
                raise ValueError(f"Checkpoint shape mismatch: {key}")
        del model, expected

        # Plan shards before writing so their final names need no renaming.
        groups, group, size, total = [], [], 0, 0
        dtypes = {}
        for key in sorted(keys):
            value = tensor(key)
            nbytes = value.numel() * value.element_size()
            dtypes[key] = str(value.dtype).removeprefix("torch.")
            if group and size + nbytes > 2_000_000_000:
                groups.append(group)
                group, size = [], 0
            group.append(key)
            size += nbytes
            total += nbytes
            del value
        if group:
            groups.append(group)
        output.mkdir(parents=True, exist_ok=True)
        weight_map = {}
        for index, group in enumerate(groups, 1):
            name = f"model-{index:05d}-of-{len(groups):05d}.safetensors"
            save_file({key: tensor(key).contiguous() for key in group}, str(output / name), metadata={"format": "pt"})
            weight_map.update({key: name for key in group})
            print(f"Saved shard {index}/{len(groups)}", flush=True)
        (output / "model.safetensors.index.json").write_text(json.dumps(
            {"metadata": {"total_size": total}, "weight_map": weight_map}, indent=2) + "\n")

    config.architectures = [Model.__name__]
    config.save_pretrained(output)
    processor.save_pretrained(output)
    (output / "reward_config.json").write_text(json.dumps(dict(
        format_version=1, reward_token="<|Reward|>", reward_token_id=token_id,
        model_kwargs=settings), indent=2) + "\n")
    print(json.dumps(dict(source_sha256=digest, tensors=len(keys), bytes=total,
                          reward_token_id=token_id, dtype_counts={d: list(dtypes.values()).count(d) for d in set(dtypes.values())})))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("project", choices=["hpsv3", "hpsv3pp"])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--expected-sha256", required=True, help="Official Hub file SHA-256")
    args = parser.parse_args()
    build(args.project, args.checkpoint, args.base_model, args.output_dir, args.expected_sha256)
