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

HPSv3's loader preserves floating-point vision inputs when the vision tower
uses NF4 weights. Transformers 4.46.3 otherwise derives the image dtype from
packed `uint8` weights, corrupting normalized pixels before both scoring and
captioning. Existing NF4 model files remain usable; update the wrapper code
to obtain this correction. Re-evaluate HPSv3 scores produced with older
wrapper code.

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

- Own repository code: MIT. The TRL shim is Apache-2.0 and the reusable
  runtime's HPSv3++ model classes have unresolved upstream permission;
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

## Canonical host runtime

`src/hpsv3_4bit` provides the inference-only runtime used by host integrations
on Transformers 5.17.x. Import `hpsv3_4bit.load_model` with family `hpsv3` or
`hpsv3pp` and a local merged NF4 directory. The older `hpsv3/` and `hpsv3pp/`
projects remain separate CLI environments and are retained as regression
baselines.

The canonical runtime does not download models, execute commands, import
training packages, or load remote code. It requires local serialized
bitsandbytes NF4 checkpoints and preserves the upstream reward protocols.

Build the standalone wheel with `uv build --wheel`. Host applications provide
their own PyTorch installation and install the wheel's declared inference
dependencies before calling the API:

```python
from PIL import Image
from hpsv3_4bit import load_model

session = load_model("hpsv3", "/local/HPSv3-bnb-NF4", device="cuda")
image = Image.open("example.png").convert("RGB")
score = session.score(image, "An image description.")
caption = session.caption(image, max_new_tokens=96)
```

Each score call evaluates one pair; HPSv3++ fixes the iteration condition at
zero. The host owns model lifetime and may pass `check_cancel` to `load_model`
and Transformers stopping criteria to `caption`. HPSv3++ code permission must
be resolved before publishing this runtime or its wheel. See the notices
inside `src/hpsv3_4bit/`.
