"""Prepare and audit an NF4 export before uploading its directory with uvx hf.

Run with the corresponding model project's Python interpreter.
Only model assets and public documentation are allowed in the export directory.
"""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import re
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
ASSETS = {
    "config.json", "generation_config.json", "model.safetensors.index.json",
    "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "chat_template.json",
    "added_tokens.json", "vocab.json", "merges.txt", "chat_template.jinja",
    "preprocessor_config.json", "video_preprocessor_config.json",
    "reward_config.json", "README.md", "LICENSE", "NOTICE", "release_manifest.json",
}


def clean_paths(value):
    if isinstance(value, dict):
        return {k: clean_paths(v) for k, v in value.items()
                if k not in {"_name_or_path", "name_or_path", "tokenizer_file", "special_tokens_map_file"}}
    if isinstance(value, list):
        return [clean_paths(v) for v in value]
    return value


def prepare(project, directory):
    directory = Path(directory)
    pp = project == "hpsv3pp"
    title = "HPSv3++" if pp else "HPSv3"
    repo = "stella221125/" + ("HPSv3-PlusPlus-bnb-NF4" if pp else "HPSv3-bnb-NF4")
    original = "Junjun2333/HPSv3-PlusPlus" if pp else "MizzenAI/HPSv3"
    revision = "ac265789b76f3d69169533713fec26cfffd92ada" if pp else "4f81e3e09edd82fe3c5f636444c721b592a735ca"
    source_file = "hpsv3++.pth" if pp else "HPSv3.safetensors"
    source_sha256 = ("f57bc6c6c9774db93d909e6c079f1472e7eeaf88a9756b5d1f52e8951d24e11f" if pp
                     else "a13d7ff5a07b7ffa0f7824e60d62e6ae144541ceefd5224b4c08fda7ab39f353")
    base = "Qwen/Qwen3-VL-8B-Instruct" if pp else "Qwen/Qwen2-VL-7B-Instruct"
    base_revision = "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b" if pp else "eed13092ef92e448dd6875b2a00151bd3f7db0ac"
    sys.path.insert(0, str(ROOT / project / "src"))
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(str(directory), local_files_only=True)
    token_id = processor.tokenizer.convert_tokens_to_ids("<|Reward|>")
    settings = dict(output_dim=2, reward_token="special", special_token_ids=[token_id],
                    rm_head_type="ranknet", rm_head_kwargs=None)
    if pp:
        settings["cond_dim"] = 256
    reward = dict(format_version=1, reward_token="<|Reward|>", reward_token_id=token_id,
                  model_kwargs=settings)
    (directory / "reward_config.json").write_text(json.dumps(reward, indent=2) + "\n", encoding="utf-8")
    for name in ("config.json", "tokenizer_config.json", "preprocessor_config.json",
                 "video_preprocessor_config.json", "generation_config.json"):
        path = directory / name
        if path.exists():
            data = clean_paths(json.loads(path.read_text(encoding="utf-8")))
            path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    config = json.loads((directory / "config.json").read_text())
    quant = config["quantization_config"]
    assert quant["load_in_4bit"] and quant["bnb_4bit_quant_type"] == "nf4"
    shutil.copyfile(ROOT / "artifacts/upstream-qwen2-license/LICENSE", directory / "LICENSE")
    notice = f"""{title} bitsandbytes NF4 derivative distribution

Official reward checkpoint: https://huggingface.co/{original}/tree/{revision}
Qwen architecture and tokenizer/processor: https://huggingface.co/{base}/tree/{base_revision}
Credit belongs to the original HPS and Qwen model authors.
The original reward-model and Qwen model cards identify their weights as Apache-2.0.
See LICENSE for the Apache License, Version 2.0.

Changes by stella221125: serialized bitsandbytes NF4 with double quantization;
normalized architecture config; bundled processor/tokenizer and reward settings;
removed local source-path metadata. No additional training or calibration.
Reward and conditioning modules are excluded from 4-bit quantization.
This is an unofficial derivative, not an endorsement by the original authors.
No upstream Python implementation is included. Code licenses are separate.
"""
    (directory / "NOTICE").write_text(notice, encoding="utf-8")
    versions = {p: importlib.metadata.version(p) for p in ("torch", "transformers", "bitsandbytes", "accelerate", "safetensors")}
    size = sum(p.stat().st_size for p in directory.glob("*.safetensors"))
    card = f"""---
license: apache-2.0
base_model:
- {original}
base_model_relation: quantized
tags:
- bitsandbytes
- nf4
- 4-bit
- reward-model
- image-quality
- human-preference
- safetensors
---

# {title} — bitsandbytes NF4

Unofficial prequantized derivative for local image/prompt preference scoring with
[hpsv3-4bit](https://github.com/Stella2211/hpsv3-4bit).
Download only **{size / 1e9:.3f} GB of weights**, instead of the BF16 source.
All model config, reward settings, tokenizer and processor files are included.
The BF16 checkpoint and base-model weight download are unnecessary for inference.

## Usage

Requires an NVIDIA CUDA GPU, approximately 9 GB free VRAM, and the corresponding
project environment (Python 3.12; see its uv.lock for exact dependencies).

```bash
git clone --recurse-submodules https://github.com/Stella2211/hpsv3-4bit
cd hpsv3-4bit
uv sync --project {project}
uv run --project {project} {project}/scripts/score_batch.py --model {repo} --input records.json --output scores.json
```

`records.json` is a list of objects with `id`, `image` (local path) and `prompt`.
`--input` also accepts a directory of images with matching `.txt` prompt files.
The project downloads the full model repository, including README, LICENSE and
NOTICE, from the latest `main` by default. Use `--revision COMMIT` to select a
specific version when needed. Cached inference supports
`HF_HUB_OFFLINE=1` and `--local-files-only`.

Use the project's custom reward-model loader. A generic AutoModel or hosted
text-generation pipeline is not a supported scorer. No remote Python code is
included; HPSv3++ still requires the project's pinned upstream submodule.

## Conversion and provenance

- Official weights: [{original}](https://huggingface.co/{original}/tree/{revision}).
- Architecture and tokenizer: [{base}](https://huggingface.co/{base}/tree/{base_revision}).
- bitsandbytes NF4, double quantization, BF16 compute; no additional training or calibration. Source revisions and SHA-256 hashes are in `release_manifest.json`.

## Limitations

NF4 scores are not guaranteed to match full BF16 inference or human preferences.
Different devices, kernels, dependency versions, prompts and batching may change
scores. {'Use iter_step=0.0 for plain preference scoring; batch composition can affect conditioning.' if pp else 'Scores use the mean output of the reward head.'}
This model inherits limitations and biases of the upstream reward and base models.

## License and privacy

Model weights are distributed under Apache-2.0; see LICENSE and NOTICE for
attribution and modifications. Repository code has separate licensing;
the upstream Python implementation is not included in this model repo.

Only model assets and public documentation are distributed. Local paths, private
evaluation inputs, per-image scores, logs and credentials are not included.
Images and prompts used for local scoring are not uploaded by this project.
Initial model retrieval contacts Hugging Face; offline cached inference does not
require contacting it.
"""
    (directory / "README.md").write_text(card, encoding="utf-8")
    audit(directory)
    hashes = {}
    for path in sorted(directory.iterdir()):
        if path.is_file() and path.name != "release_manifest.json":
            with path.open("rb") as stream:
                hashes[path.name] = hashlib.file_digest(stream, "sha256").hexdigest()
    manifest = dict(format_version=1, repo_id=repo, source_repo=original,
                    source_revision=revision, source_file=source_file,
                    source_sha256=source_sha256, base_repo=base, base_revision=base_revision,
                    build_versions=versions, sha256=hashes)
    (directory / "release_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"RELEASE_READY {repo}: {len(hashes) + 1} files, {size / 1e9:.3f} GB weights")


def audit(directory):
    from safetensors import safe_open
    for path in directory.iterdir():
        if path.name == ".cache" and path.is_dir():
            continue  # HF upload bookkeeping; explicitly excluded from upload.
        if not path.is_file() or (path.name not in ASSETS and not re.fullmatch(r"model-\d{5}-of-\d{5}\.safetensors", path.name)):
            raise ValueError(f"Unexpected release file: {path.name}")
        if path.suffix == ".safetensors":
            with safe_open(str(path), framework="pt", device="cpu") as weights:
                if weights.metadata() != {"format": "pt"}:
                    raise ValueError(f"Unexpected tensor metadata: {path.name}")
        elif path.name not in {"tokenizer.json", "vocab.json", "merges.txt"}:
            text = path.read_text(encoding="utf-8")
            if re.search(r"(?i)((?<![a-z])[a-z]:[\\/]|/home/|/Users/|hf_[A-Za-z0-9]{20,})", text):
                raise ValueError(f"Possible private metadata in {path.name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("project", choices=["hpsv3", "hpsv3pp"])
    parser.add_argument("directory")
    args = parser.parse_args()
    prepare(args.project, args.directory)
