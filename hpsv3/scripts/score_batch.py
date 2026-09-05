"""Batch HPSv3 scoring CLI.

``--input`` accepts either:

- a JSON file: a list of ``{"id": ..., "image": <path>, "prompt": <text>}``
  records, or
- a directory: every image in it (png/jpg/jpeg/webp, case-insensitive) is
  scored, with the prompt read from a same-named text file next to it
  (``image.png`` -> ``image.txt``; extension configurable via
  ``--prompt-ext``). Records get ``id`` = image file name.

With ``--no-prompt``, prompt files / JSON prompts are not required: a short
caption is generated for each image by the loaded Qwen VL backbone itself
and used as the scoring prompt (saved to the output as
``generated_prompt``).

The script loads the 4-bit HPSv3 model once, scores every record in batched
passes through the merged checkpoint, and writes ``{"id": ..., "score":
<float>}`` results plus timing/VRAM metadata as JSON.

Usage (from the repository root):
    CUDA_VISIBLE_DEVICES=0 uv run --project hpsv3 hpsv3/scripts/score_batch.py \
        --merged-dir /path/to/hpsv3-merged-bf16 \
        --input records.json --output scores.json

Use CUDA_VISIBLE_DEVICES to pick the GPU. HPSv3 in 4-bit needs ~8.7GB peak
VRAM, so the chosen GPU should be otherwise idle on a 12GB card.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}


def positive_int(value: str) -> int:
    ivalue = int(value)
    if ivalue <= 0:
        raise argparse.ArgumentTypeError(f"must be a positive integer, got {value!r}")
    return ivalue


def prompt_ext(value: str) -> str:
    ext = value if value.startswith(".") else "." + value
    if ext.lower() in IMAGE_EXTS:
        raise argparse.ArgumentTypeError(
            f"must not be an image extension ({', '.join(sorted(IMAGE_EXTS))}), got {value!r}"
        )
    return ext


def load_records(input_path: str, prompt_ext: str = ".txt", no_prompt: bool = False) -> list[dict]:
    """Build scoring records from `input_path`.

    If it is a directory: enumerate contained images (IMAGE_EXTS,
    case-insensitive, sorted by name) and read each image's prompt from the
    same-named `prompt_ext` file (unless `no_prompt`); missing prompt files
    are reported all at once and abort. If it is a file: parse it as the
    JSON record list.
    """
    path = Path(input_path)
    if path.is_dir():
        if not prompt_ext.startswith("."):
            prompt_ext = "." + prompt_ext
        images = sorted(p for p in path.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
        if not no_prompt:
            missing = [p.name for p in images if not p.with_suffix(prompt_ext).is_file()]
            if missing:
                raise SystemExit(
                    f"error: no {prompt_ext} prompt file found for {len(missing)} image(s) in "
                    f"{path}: {', '.join(missing)}\n"
                    "Add the missing prompt files, or pass --no-prompt to auto-generate captions."
                )
        records = []
        for p in images:
            record = {"id": p.name, "image": str(p)}
            if not no_prompt:
                record["prompt"] = p.with_suffix(prompt_ext).read_text(encoding="utf-8").strip()
            records.append(record)
        return records

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--input",
        required=True,
        help="JSON list of {id, image, prompt} records, or a directory of images "
        "with same-named prompt text files",
    )
    parser.add_argument("--output", required=True, help="Path to write JSON results")
    parser.add_argument("--merged-dir", required=True, help="Path to the merged bf16 HPSv3 checkpoint")
    parser.add_argument(
        "--processor-dir",
        default=None,
        help="Where to load the tokenizer/processor from (default: --merged-dir if it "
        "contains processor files, otherwise the Qwen/Qwen2-VL-7B-Instruct base model)",
    )
    parser.add_argument(
        "--batch-size", type=positive_int, default=4, help="Images scored per forward pass (VRAM/time tradeoff)"
    )
    parser.add_argument(
        "--prompt-ext",
        type=prompt_ext,
        default=".txt",
        help="Extension of per-image prompt files in directory input mode (default: .txt)",
    )
    parser.add_argument(
        "--no-prompt",
        action="store_true",
        help="Don't read prompts; instead generate a caption for each image with the "
        "loaded Qwen VL backbone and score against it (saved as generated_prompt)",
    )
    args = parser.parse_args()

    records = load_records(args.input, prompt_ext=args.prompt_ext, no_prompt=args.no_prompt)
    if not records:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump({"scores": [], "load_time_sec": 0.0, "infer_time_sec": 0.0, "peak_vram_gb": 0.0}, f)
        return

    import torch

    from src.evaluation.hpsv3_quantized import HPSv3QuantizedInferencer

    assert torch.cuda.is_available(), "CUDA not visible -- check CUDA_VISIBLE_DEVICES"
    torch.zeros(1, device="cuda:0")
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats(0)

    t0 = time.time()
    inferencer = HPSv3QuantizedInferencer.from_merged_dir(
        args.merged_dir, device="cuda:0", processor_dir=args.processor_dir
    )
    load_time = time.time() - t0
    load_peak_gb = torch.cuda.max_memory_allocated(0) / 1e9
    print(f"Loaded 4-bit HPSv3 in {load_time:.1f}s, peak VRAM after load: {load_peak_gb:.2f} GB")

    torch.cuda.reset_peak_memory_stats(0)
    t0 = time.time()

    results = []
    for i in range(0, len(records), args.batch_size):
        chunk = records[i : i + args.batch_size]
        chunk_images = [r["image"] for r in chunk]
        if args.no_prompt:
            chunk_prompts = inferencer.caption(chunk_images)
        else:
            chunk_prompts = [r["prompt"] for r in chunk]
        chunk_scores = inferencer.score(chunk_images, chunk_prompts)
        for r, p, s in zip(chunk, chunk_prompts, chunk_scores):
            row = {"id": r["id"], "score": s}
            if args.no_prompt:
                row["generated_prompt"] = p
            results.append(row)

    infer_time = time.time() - t0
    infer_peak_gb = torch.cuda.max_memory_allocated(0) / 1e9
    overall_peak_gb = max(load_peak_gb, infer_peak_gb)
    print(f"Scored {len(records)} images in {infer_time:.2f}s, peak VRAM during inference: {infer_peak_gb:.2f} GB")

    result = {
        "scores": results,
        "load_time_sec": load_time,
        "infer_time_sec": infer_time,
        "peak_vram_gb": overall_peak_gb,
    }
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
