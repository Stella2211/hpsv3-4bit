# hpsv3-4bit

Run HPSv3 and HPSv3++ image/prompt preference scoring locally on a single
12 GB NVIDIA GPU with bitsandbytes NF4 4-bit models.

Tested on 12 GB VRAM. An 8 GB GPU may also work with smaller batches, but
8 GB operation has not been verified. Actual memory use depends on image
resolution, batch size and runtime overhead.

The first run automatically downloads the published quantized model from
Hugging Face; subsequent runs use the cache.

| Scorer | Backbone | Default model | Weight download |
|---|---|---|---:|
| HPSv3 | Qwen2-VL-7B | [stella221125/HPSv3-bnb-NF4](https://huggingface.co/stella221125/HPSv3-bnb-NF4) | 5.908 GB |
| HPSv3++ | Qwen3-VL-8B | [stella221125/HPSv3-PlusPlus-bnb-NF4](https://huggingface.co/stella221125/HPSv3-PlusPlus-bnb-NF4) | 6.454 GB |

## Requirements

- NVIDIA GPU with CUDA support. A 12 GB card provides room for model weights
  and inference; memory use depends on image resolution and batch size.
- An NVIDIA driver compatible with CUDA 13.0, the PyTorch wheel index
  configured in both projects.
- Python 3.12 or newer and [uv](https://docs.astral.sh/uv/).
- About 8 GB disk space per model, plus Python environments and download
  temporary space.

## Install

```bash
git clone --recurse-submodules https://github.com/Stella2211/hpsv3-4bit
cd hpsv3-4bit
```

Install the scorer you want to use (or run both commands for both scorers):

```bash
uv sync --project hpsv3
uv sync --project hpsv3pp
```

The projects use separate environments because they require different
Transformers versions: 4.46.3 for HPSv3 and 4.57.0 for HPSv3++.
HPSv3++ also requires the pinned upstream submodule. If you cloned without
submodules, run `git submodule update --init`.

## Score images

Run commands from the repository root. Create `records.json` with image/prompt pairs:

```json
[
  {"id": "cat", "image": "images/cat.png", "prompt": "a cat sitting by a window"},
  {"id": "dog", "image": "images/dog.png", "prompt": "a dog playing in a park"}
]
```

Relative image paths are resolved from the current working directory.
Run either scorer:

```bash
uv run --project hpsv3 hpsv3/scripts/score_batch.py --input records.json --output scores.json
uv run --project hpsv3pp hpsv3pp/scripts/score_batch.py --input records.json --output scores.json
```

Each command writes a JSON object with a `scores` list of `id`/`score` pairs,
plus load time, inference time and peak GPU memory measurements. Choose
separate output paths to keep results from both models.

### Image directories

`--input` also accepts a directory of `.png`, `.jpg`, `.jpeg` or `.webp`
images (case-insensitive, non-recursive). Each image needs a matching prompt
file: `image.png` uses `image.txt`. Missing prompt files abort the run.
Use `--prompt-ext` to change the text-file extension. Output IDs are image
file names.

```bash
uv run --project hpsv3pp hpsv3pp/scripts/score_batch.py --input images --output scores.json
```

### Images without prompts

Add `--no-prompt` to generate a caption for each image and use it as the
scoring prompt. This works with directory and JSON input; prompt files or
JSON `prompt` fields are then unnecessary. Captions are saved as
`generated_prompt` in the output. Review them: the captions come from the
reward-finetuned model and may misdescribe an image.

```bash
uv run --project hpsv3pp hpsv3pp/scripts/score_batch.py --input images --no-prompt --output scores.json
```

### Batch size and conditioning

`--batch-size` defaults to 4 for HPSv3 and 2 for HPSv3++. Reduce it if you run
out of GPU memory. Use `CUDA_VISIBLE_DEVICES` to select a GPU.

HPSv3++ accepts `--iter-step` in [0, 1], defaulting to `0.0` for plain
preference scoring. Its conditioning can depend on other images in the
batch, so keep batch size, image order, prompts and iteration setting fixed
when comparing scores.

## Model selection and offline use

Both scorers accept:

| Option | Purpose |
|---|---|
| `--model PATH_OR_REPO` | Load a local checkpoint or another compatible Hugging Face model repository. |
| `--revision REVISION` | Select a Hub branch, tag or commit; the default is `main`. |
| `--local-files-only` | Use local or cached files; fail if required files are missing. |
| `--processor-dir PATH_OR_REPO` | Override the tokenizer/processor bundled with the model. |

HPSv3++ requires the model's `config.json` to contain its Qwen3-VL
`text_config` and `vision_config`. Legacy community exports containing only
a training config are rejected. Use the published NF4 model or build a complete
checkpoint with `hpsv3pp/scripts/merge_bf16.py`; architecture settings are not
bundled in this code repository.

## Python API

Run each example in its corresponding project environment, from the repository root.

HPSv3 (`uv run --project hpsv3 python`):

```python
import sys
sys.path.insert(0, "hpsv3/src")
from evaluation.hpsv3_quantized import HPSv3QuantizedInferencer

scorer = HPSv3QuantizedInferencer.from_merged_dir()
scores = scorer.score(["images/cat.png"], ["a cat sitting by a window"])
```

HPSv3++ (`uv run --project hpsv3pp python`):

```python
import sys
sys.path.insert(0, "hpsv3pp/src")
from evaluation.hpsv3pp_quantized import HPSv3PPQuantizedInferencer

scorer = HPSv3PPQuantizedInferencer.from_merged_dir()
scores = scorer.score(["images/cat.png"], ["a cat sitting by a window"], iter_step=0.0)
```

`from_merged_dir()` accepts an optional local path or Hub repository ID,
plus `revision=` and `local_files_only=`. Use these custom reward-model
classes for scoring; generic `AutoModel` loading does not provide this interface.

## Quantization

The published models use bitsandbytes NF4 with double quantization. Reward
heads and conditioning modules are excluded from 4-bit quantization;
excluded modules retain floating-point weights, which does not imply FP32.

In our comparison, GPTQ had comparable quality to bnb 4-bit but used more VRAM.

## Licensing notes

- Repository code: MIT, except the TRL compatibility shim (Apache-2.0);
  see [LICENSE](LICENSE) and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
- The HPSv3 model class adapts MizzenAI/HPSv3 code. Its MIT attribution and
  license are preserved in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
- HPSv3++ upstream code is referenced as a pinned submodule and has no
  LICENSE file in that checkout. Its weight license does not grant a license
  to the Python implementation.
- The published NF4 weights carry Apache-2.0 separately from this repository's
  code license. Each model repository includes LICENSE, NOTICE, source
  attribution and conversion settings. Model weights are not stored here.

## Acknowledgements

- [HPSv3](https://github.com/MizzenAI/HPSv3) — MizzenAI
- [HPSv3++](https://github.com/PlantPotatoOnMoon/HPSv3-PlusPlus)
- [Qwen2-VL](https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct) and
  [Qwen3-VL](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct) — Alibaba Qwen team
- [bitsandbytes](https://github.com/bitsandbytes-foundation/bitsandbytes)
