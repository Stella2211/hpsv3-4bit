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
  configured in the root inference project.
- Python 3.12 or newer and [uv](https://docs.astral.sh/uv/).
- About 8 GB disk space per model, plus Python environments and download
  temporary space.

## Install

```bash
git clone https://github.com/Stella2211/hpsv3-4bit
cd hpsv3-4bit
```

Install both scorers into one inference environment:

```bash
uv sync
```

Both scorers now use the canonical `hpsv3_4bit` package on Transformers
5.17.x. Scoring does not require the nested upstream submodule or training
dependencies. Existing model downloads remain usable.

When updating an older checkout, run `uv sync` at the repository root and
replace `uv run --project hpsv3 hpsv3/scripts/score_batch.py` with
`uv run hpsv3-score` (and likewise `hpsv3pp-score`). CLI options and
result JSON fields are preserved. Old script paths are thin entry points
and can still be run with `uv run python hpsv3/scripts/score_batch.py` or
`uv run python hpsv3pp/scripts/score_batch.py` in the root environment.
The former project environments are now for conversion only.

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
uv run hpsv3-score --input records.json --output scores.json
uv run hpsv3pp-score --input records.json --output scores.json
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
uv run hpsv3pp-score --input images --output scores.json
```

### Images without prompts

Add `--no-prompt` to generate a caption for each image and use it as the
scoring prompt. This works with directory and JSON input; prompt files or
JSON `prompt` fields are then unnecessary. Captions are saved as
`generated_prompt` in the output. Review them: the captions come from the
reward-finetuned model and may misdescribe an image.

```bash
uv run hpsv3pp-score --input images --no-prompt --output scores.json
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

Run `uv run python` from the repository root:

```python
from PIL import Image
from hpsv3_4bit import load_model

session = load_model("hpsv3pp", "/local/HPSv3-PlusPlus-bnb-NF4")
image = Image.open("images/cat.png").convert("RGB")
score = session.score(image, "a cat sitting by a window")
caption = session.caption(image)
scores = session.score_batch([image], ["a cat sitting by a window"], iter_step=0.0)
```

Use family `"hpsv3"` for HPSv3. `score` returns one float and `caption`
returns one string. `score_batch` returns a list using a single batched
forward; it preserves CLI batch composition and HPSv3++ conditioning.
For independent per-image scores, call `score` for each pair.

The former `evaluation.hpsv3_quantized` and `evaluation.hpsv3pp_quantized`
inference APIs have been removed. Replace their list-based calls with
`score_batch` and single-image `caption` calls. The public loader accepts
local directories only. Applications needing the CLI's Hub resolution can
explicitly resolve files before loading:

```python
from hpsv3_4bit.model_source import resolve_model_source

directory, processor_directory = resolve_model_source(
    "hpsv3pp", revision=None, local_files_only=False,
)
session = load_model("hpsv3pp", directory, processor_directory=processor_directory)
```

Model acquisition stays outside inference; `load_model` never downloads.
Both the new API and CLIs require serialized NF4 checkpoints. If you used
the old loader to quantize a BF16 directory on the fly, first run the
corresponding `export_bnb.py` in its conversion environment and pass the
exported directory to the new scorer.

## Quantization

The published models use bitsandbytes NF4 with double quantization. Reward
heads and conditioning modules are excluded from 4-bit quantization;
excluded modules retain floating-point weights, which does not imply FP32.

In our comparison, GPTQ had comparable quality to bnb 4-bit but used more VRAM.

BF16 merging and NF4 export retain their separate conversion environments:

```bash
git submodule update --init --recursive
uv sync --project hpsv3
uv sync --project hpsv3pp
```

Run conversion scripts with their existing `uv run --project hpsv3 ...` or
`uv run --project hpsv3pp ...` commands. Those projects retain Transformers
4.46.3 and 4.57.0 respectively for the original checkpoint construction
paths. They are not inference environments. Their conversion-only model
definitions do not implement CLI Score or Caption.

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

`src/hpsv3_4bit` provides the shared runtime used by both CLIs and host integrations
on Transformers 5.17.x. Import `hpsv3_4bit.load_model` with family `hpsv3` or
`hpsv3pp` and a local merged NF4 directory. All NF4 loading, Score and Caption
implementations are maintained here; CLI modules only resolve files and
adapt command-line input/output.

The inference API does not download models, execute commands, import
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

Each `score` call evaluates one pair with iteration condition zero.
`score_batch` also supports explicit HPSv3++ iteration conditioning.
The host owns model lifetime and may pass `check_cancel` to `load_model`
and Transformers stopping criteria to `caption`. HPSv3++ code permission must
be resolved before publishing this runtime or its wheel. See the notices
inside `src/hpsv3_4bit/`.
