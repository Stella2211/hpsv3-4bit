# Development guide

## Project structure

- `hpsv3/`: HPSv3 conversion tools, pinned to Transformers 4.46.3.
- `hpsv3pp/`: HPSv3++ conversion tools, pinned to Transformers 4.57.0.
- `src/hpsv3_4bit/`: canonical inference-only runtime for host Transformers
  5.17.x. It keeps both families under distinct namespaces and loads only
  prepared local merged NF4 checkpoints.
- The root project installs both scoring CLIs and the canonical runtime.
  Each conversion project retains its own `pyproject.toml`, `uv.lock`,
  conversion helpers and tests. Old score script paths are thin entry points
  into the root package, not separate inference implementations.
- HPSv3++ compatibility source is prepared from the pinned upstream revision
  on the first `hpsv3pp-score` invocation or by the explicit `ensure_source`
  API; the runtime uses a thin adapter and does not redistribute the upstream
  checkout. Preserve upstream attribution and separate code/weight licenses.
- Root `scripts/` contains NF4 release preparation and validation tools;
  root `tests/` covers the shared Hub-loading contract.

## Tools and setup

Use Python 3.12 or newer and uv. Run commands from the repository root:

```bash
uv sync
uv run hpsv3-score --help
uv run hpsv3pp-score --help
```

Use the root environment for inference. For BF16 merge/NF4 export only,
use `uv sync --project hpsv3` or
`uv sync --project hpsv3pp`, then `uv run --project <family> ...`.
Keep dependency changes and the corresponding lockfile in sync.
The projects select CUDA 13.0 PyTorch wheels; GPU inference requires
a compatible NVIDIA GPU and driver. CPU tests do not require model downloads.

## Implementation conventions

- Follow nearby Python code: four-space indentation, snake_case functions and
  variables, PascalCase classes, and type hints on new public interfaces.
- Keep changes focused. Do not reformat unrelated files or change dependencies
  solely to resolve formatting differences. No formatter/linter is configured.
- Keep both scorers' CLI options and Hub-loading behavior in the shared CLI.
  All NF4 inference and compatibility fixes belong in `src/hpsv3_4bit/`.
  Do not restore duplicate scoring or captioning implementations under the
  conversion projects. Conversion model definitions remain separately pinned
  only where checkpoint construction requires the old Transformers layout.
- Default inference loads the published NF4 model automatically. Preserve local
  paths, Hub IDs, explicit revisions, cached/offline loading, and the
  `--merged-dir` compatibility alias for `--model`.
- Reuse serialized NF4 quantization settings. Apply original reward checkpoints
  to full-precision models before quantizing; packed weights cannot accept a
  full-precision state dict. Validate reward-token IDs and architecture config
  against the tokenizer and checkpoint rather than silently guessing.
- Exclude reward/conditioning modules from 4-bit quantization. Quantization
  exclusion does not guarantee FP32; avoid accidental dtype changes.
- HPSv3++ scores can depend on batch composition. Keep image order, prompts,
  batch size and `iter_step` fixed in score comparisons.
- Document public behavior and non-obvious constraints. Do not put session logs,
  transient benchmark results, machine-specific paths, or abandoned approaches
  in README, comments, or this file. Update usage examples when behavior changes.

## Validation

Run the relevant existing tests after implementation changes:

```bash
uv run --no-project --python <host-python> python -m unittest discover -s tests -p test_runtime_api.py -v
uv run python -m unittest discover -s tests -p test_cli.py -v
uv run --project hpsv3 python -m unittest discover -s tests -p test_hub_loading.py -v
uv run --project hpsv3 python -m unittest discover -s hpsv3/tests -v
uv run --project hpsv3pp python -m unittest discover -s hpsv3pp/tests -v
```

Check CLI changes with each scorer's `--help`. Add focused regression tests for
behavioral fixes; documentation-only changes need link/example review, not new tests.

For loader or inference changes, run a GPU smoke test in the affected environment:

```bash
uv run python scripts/validate_release.py hpsv3pp --output artifacts/hpsv3pp-smoke.json
```

Create `artifacts/` first if absent. Substitute `hpsv3` for HPSv3 validation.
The script scores synthetic images, verifies finite scores and a nonempty caption,
and reports peak allocated VRAM. Use `--local-files-only` with a populated cache;
`HF_HOME` selects the cache location and `HF_HUB_OFFLINE=1` disables Hub access.
Use `--baseline REPORT.json` to compare scores and captions with a prior report
from the same inputs. GPU allocation statistics exclude CUDA context and other
processes, so they are not a guarantee that a model fits a particular card.

## Repository hygiene

- Keep model weights, datasets, scores, generated reports, logs and caches out
  of Git. Put local validation outputs under ignored `artifacts/`.
- Do not modify model assets or private evaluation data as part of code cleanup.
- Before committing, inspect `git status`, the staged diff and `git diff --check`.
  Report what was tested and any checks that could not run.
- Model publishing is separate from code changes. The preparation script embeds
  source revisions: review them for each release and upload only audited model
  assets. Do not publish or upload without explicit authorization.
