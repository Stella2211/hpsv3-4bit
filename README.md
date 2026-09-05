# hpsv3-4bit

Run the [HPSv3](https://github.com/MizzenAI/HPSv3) and
[HPSv3++](https://github.com/PlantPotatoOnMoon/HPSv3-PlusPlus) image-quality
reward models on a single 12GB GPU (e.g. RTX 3060) using bitsandbytes NF4
4-bit quantization.

- **HPSv3** (Qwen2-VL-7B backbone) — ~8.7GB peak VRAM
- **HPSv3++** (Qwen3-VL-8B backbone) — **7.11GB peak VRAM** (6.64GB after
  load), measured on an RTX 3060 12GB

Licensing, briefly (details under [Licensing notes](#licensing-notes-important)):

- **Code**: this repository is MIT; the HPSv3 code it adapts
  (MizzenAI/HPSv3) is also MIT. The HPSv3++ code repository has no license
  file and is only referenced as a submodule, not redistributed.
- **Base models**: Qwen2-VL-7B-Instruct and Qwen3-VL-8B-Instruct are
  Apache-2.0.
- **Fine-tuned weights**: not redistributed here; downloaded from Hugging
  Face under the license stated on each model card.

## Why this exists

Both models ship as full-precision checkpoints (~17GB), which the upstream
code loads into a full-precision skeleton via `load_state_dict(strict=True)`.
That is incompatible with loading directly under bitsandbytes quantization:
packed 4-bit weights have different shapes than the full-precision state
dict, so quantize-then-apply fails with shape mismatches. This repo
implements the two-stage workaround:

1. **One-time CPU-only merge**: build a bf16 skeleton, apply the released
   checkpoint with strict shape checking, `save_pretrained()` to disk.
2. **4-bit reload**: load that merged dir with
   `BitsAndBytesConfig(load_in_4bit=True, nf4, double-quant)`, which
   quantizes weights as they stream from disk.

The reward head and conditioning modules stay in fp32, matching upstream.
(For both models a community pre-merged bf16 safetensors export exists, so
step 1 can usually be skipped — see Usage.)

## Requirements

- NVIDIA GPU with ~9GB free VRAM (HPSv3) / ~8GB (HPSv3++). CUDA 12.4 wheels
  are pinned via the `pytorch-cu124` index in each `pyproject.toml`; edit
  that index for other CUDA versions.
- ~40GB free disk per model (HF cache + merged bf16 copy) and ~40GB host RAM
  for the merge step (CPU-only).
- [uv](https://docs.astral.sh/uv/), Python 3.12.

## Layout

Two independent uv projects — they cannot share a venv, because HPSv3 needs
`transformers==4.46.3` (the last release with the old flat Qwen2VL module
layout its checkpoint expects) while HPSv3++ needs `transformers==4.57.0`
(Qwen3-VL support):

- `hpsv3/` … HPSv3
- `hpsv3pp/` … HPSv3++ (+ upstream code as a pinned git submodule under
  `hpsv3pp/third_party/HPSv3-PlusPlus`)

There is deliberately no root `pyproject.toml` / uv workspace: a shared lock
would force the two transformers pins to conflict.

Note: `transformers==4.57.0` is yanked on PyPI (packaging issue), but an
exact pin still installs fine with `uv sync` — no functional impact observed.

## Install

```bash
git clone --recurse-submodules https://github.com/Stella2211/hpsv3-4bit
cd hpsv3-4bit/hpsv3   && uv sync
cd ../hpsv3pp         && uv sync
```

(If you cloned without submodules: `git submodule update --init`.)

## Usage — HPSv3

Recommended: download the pre-merged bf16 export
([sitatech/HPSv3](https://huggingface.co/sitatech/HPSv3), a full merged
export with tokenizer/processor files included) and load it in 4-bit
directly — no merge step needed:

```bash
huggingface-cli download sitatech/HPSv3 --local-dir /path/to/hpsv3-bf16
CUDA_VISIBLE_DEVICES=0 uv run --project hpsv3 hpsv3/scripts/score_batch.py \
    --merged-dir /path/to/hpsv3-bf16 \
    --input records.json --output scores.json
# records.json: [{"id": ..., "image": "/path/img.png", "prompt": "..."}, ...]
# (relative image paths are resolved against the current working directory)
```

Note: sitatech/HPSv3 is a community re-upload without a license tag on its
model card; the original HPSv3 weights it derives from
([MizzenAI/HPSv3](https://huggingface.co/MizzenAI/HPSv3)) are Apache-2.0.

Alternative: merge the official checkpoint yourself (CPU-only; downloads
Qwen2-VL-7B + `HPSv3.safetensors`), then point `--merged-dir` at the output:

```bash
uv run --project hpsv3 hpsv3/scripts/merge_bf16.py \
    --output-dir /path/to/hpsv3-merged-bf16
```

There is also `hpsv3/scripts/smoke_test.py` for a quick check on a couple of
images (`--image ... --prompt ...`, repeatable).

## Usage — HPSv3++

Recommended: download the pre-merged bf16 safetensors export
([bdsqlsz/HPSV3-PlusPLus-BF16](https://huggingface.co/bdsqlsz/HPSV3-PlusPLus-BF16),
Apache-2.0) and load it in 4-bit directly — no merge step needed:

```bash
huggingface-cli download bdsqlsz/HPSV3-PlusPLus-BF16 --local-dir /path/to/hpsv3pp-bf16
CUDA_VISIBLE_DEVICES=0 uv run --project hpsv3pp hpsv3pp/scripts/score_batch.py \
    --merged-dir /path/to/hpsv3pp-bf16 --input records.json --output scores.json
```

Note: the bdsqlsz export contains only `config.json` and the weight
safetensors — no tokenizer/processor files. The scripts handle this
automatically: when the merged dir has no processor files, the processor is
loaded from the base model (`Qwen/Qwen3-VL-8B-Instruct`, downloaded from the
Hub) and the reward token is re-added. Pass `--processor-dir` (or the
`processor_dir=` argument in the Python API) to load it from somewhere else.

`score_batch.py` supports `--iter-step` (HPSv3++'s normalized RL-iteration
conditioning value in [0, 1]; default 0.0 = plain preference scoring, as
recommended upstream).

Alternative: merge the official checkpoint
([Junjun2333/HPSv3-PlusPlus](https://huggingface.co/Junjun2333/HPSv3-PlusPlus),
`hpsv3++.pth`) yourself — CPU-only, ~40GB RAM — then point `--merged-dir` at
the output:

```bash
uv run --project hpsv3pp hpsv3pp/scripts/merge_bf16.py \
    --output-dir /path/to/hpsv3pp-merged-bf16
```

## Input modes (both scorers)

`--input` for both `score_batch.py` scripts accepts either a JSON file or a
directory:

- **JSON file**: a list of `{"id": ..., "image": "/path/img.png",
  "prompt": "..."}` records (as in the examples above).
- **Directory**: every image in the directory (`.png`/`.jpg`/`.jpeg`/`.webp`,
  case-insensitive) is scored, and each image's prompt is read from a
  same-named text file next to it (`image.png` → `image.txt`). The text file
  extension can be changed with `--prompt-ext` (default `.txt`). Images
  without a matching prompt file are listed and the run aborts. Output
  records use the image file name as `id`.

```bash
CUDA_VISIBLE_DEVICES=0 uv run --project hpsv3 hpsv3/scripts/score_batch.py \
    --merged-dir /path/to/hpsv3-bf16 \
    --input /path/to/image_dir --output scores.json
```

With `--no-prompt`, no prompts are needed at all (in either mode): the
loaded Qwen VL backbone itself first generates a one-sentence caption for
each image, which is then used as the scoring prompt and saved to the
output record as `generated_prompt`. This is a convenience for scoring
unlabeled image sets; the captions come from the reward-finetuned model's
language head, so review the saved `generated_prompt` values if scores look
surprising.

## Python API

```python
# inside the hpsv3 project (uv run --project hpsv3 python ...)
import sys; sys.path.insert(0, "hpsv3")
from src.evaluation.hpsv3_quantized import HPSv3QuantizedInferencer

inf = HPSv3QuantizedInferencer.from_merged_dir("/path/to/hpsv3-merged-bf16")
scores = inf.score(["img.png"], ["a photo of ..."])
```

`HPSv3PPQuantizedInferencer` in `hpsv3pp/src/evaluation/hpsv3pp_quantized.py`
has the same interface (plus an `iter_step` argument on `score()`).

## Performance (RTX 3060 12GB)

| model   | load time | peak VRAM (load) | peak VRAM (inference) |
|---------|-----------|------------------|-----------------------|
| HPSv3   | ~58s      | 6.10GB           | 8.66GB                |
| HPSv3++ | ~60s      | 6.64GB           | 7.11GB                |

Measurement notes: VRAM figures are `torch.cuda.max_memory_allocated()`
(PyTorch tensor allocations only — the CUDA context/driver overhead and
allocator-reserved-but-unused memory are not included, so `nvidia-smi` will
report more). Measured at batch size 4 with the pinned CUDA 12.4 / PyTorch
wheels; exact numbers vary with batch size, image resolution, and
CUDA/PyTorch versions.

## Licensing notes (IMPORTANT)

- Code in this repository: MIT (see `LICENSE`).
- `hpsv3/src/evaluation/hpsv3_model.py` contains a model class adapted from
  [MizzenAI/HPSv3](https://github.com/MizzenAI/HPSv3) (MIT); attribution is
  preserved in the file header, and the upstream copyright notice and full
  MIT license text are reproduced in `THIRD_PARTY_NOTICES.md`.
- **The HPSv3++ code repository has no LICENSE file** on GitHub. For that
  reason its code is referenced only as a pinned git submodule and is NOT
  redistributed here; whether and how you use that code is your own decision
  — review the upstream repository. (The HPSv3++ *weights* on Hugging Face —
  both Junjun2333/HPSv3-PlusPlus and the pre-merged
  bdsqlsz/HPSV3-PlusPLus-BF16 — are published under Apache-2.0.)
- Model weights are NOT redistributed; they are downloaded from the original
  Hugging Face repositories under their own licenses.

## Acknowledgements

- [HPSv3: Towards Wide-Spectrum Human Preference Score](https://github.com/MizzenAI/HPSv3) (MizzenAI)
- [HPSv3++](https://github.com/PlantPotatoOnMoon/HPSv3-PlusPlus) (arXiv:2606.14657)
- [Qwen2-VL](https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct) / [Qwen3-VL](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct) (Alibaba Qwen team)
- [bitsandbytes](https://github.com/bitsandbytes-foundation/bitsandbytes)
