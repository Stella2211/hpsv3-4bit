"""Score image/prompt pairs with HPSv3++ and write scores plus timing/VRAM JSON.

The published NF4 model downloads automatically on first use.
--input accepts a JSON list of {id, image, prompt} records or a directory
of images with matching prompt files (image.png -> image.txt).
--no-prompt generates captions and saves them as generated_prompt.

Usage (from the repository root):
    uv run --project hpsv3pp hpsv3pp/scripts/score_batch.py --input records.json --output scores.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

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


def unit_interval(value: str) -> float:
    fvalue = float(value)
    if not 0.0 <= fvalue <= 1.0:
        raise argparse.ArgumentTypeError(f"must be in [0, 1], got {value!r}")
    return fvalue


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
    parser.add_argument("--model", "--merged-dir", dest="merged_dir", help="Local checkpoint or Hub repo ID; defaults to the published bnb NF4 model")
    parser.add_argument("--revision", help="Hub branch, tag or commit (default: main)")
    parser.add_argument("--local-files-only", action="store_true", help="Use only cached/local model files")
    parser.add_argument(
        "--processor-dir",
        default=None,
        help="Where to load the tokenizer/processor from (default: selected model if it "
        "contains processor files, otherwise the Qwen/Qwen3-VL-8B-Instruct base model)",
    )
    parser.add_argument(
        "--batch-size", type=positive_int, default=2, help="Images scored per forward pass (VRAM/time tradeoff)"
    )
    parser.add_argument(
        "--iter-step",
        type=unit_interval,
        default=0.0,
        help="HPSv3++ conditioning value in [0, 1] (normalized RL-iteration condition); "
        "0.0 = plain preference scoring, as recommended upstream",
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
        with open(args.output, "w") as f:
            json.dump(
                {
                    "scores": [],
                    "load_time_sec": 0.0,
                    "load_peak_vram_gb": 0.0,
                    "infer_time_sec": 0.0,
                    "infer_peak_vram_gb": 0.0,
                },
                f,
                indent=2,
            )
        return

    import torch

    from evaluation.hpsv3pp_quantized import HPSv3PPQuantizedInferencer

    assert torch.cuda.is_available(), "CUDA not visible -- check CUDA_VISIBLE_DEVICES"
    torch.zeros(1, device="cuda:0")
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats(0)

    t0 = time.time()
    inferencer = HPSv3PPQuantizedInferencer.from_merged_dir(
        args.merged_dir, device="cuda:0", processor_dir=args.processor_dir,
        revision=args.revision, local_files_only=args.local_files_only
    )
    torch.cuda.synchronize()
    load_time = time.time() - t0
    load_peak_gb = torch.cuda.max_memory_allocated(0) / 1e9
    print(f"Loaded 4-bit HPSv3++ in {load_time:.1f}s, peak VRAM after load: {load_peak_gb:.2f} GB")

    torch.cuda.reset_peak_memory_stats(0)
    t0 = time.time()

    results = []
    bs = args.batch_size
    for i in range(0, len(records), bs):
        chunk = records[i : i + bs]
        images = [r["image"] for r in chunk]
        if args.no_prompt:
            prompts = inferencer.caption(images)
        else:
            prompts = [r["prompt"] for r in chunk]
        scores = inferencer.score(images, prompts, iter_step=args.iter_step)
        for r, p, s in zip(chunk, prompts, scores):
            row = {"id": r["id"], "score": s}
            if args.no_prompt:
                row["generated_prompt"] = p
            results.append(row)
        print(f"  scored {i + len(chunk)}/{len(records)}", flush=True)

    torch.cuda.synchronize()
    infer_time = time.time() - t0
    peak_vram_gb = torch.cuda.max_memory_allocated(0) / 1e9
    print(f"Scored {len(records)} records in {infer_time:.1f}s, peak VRAM: {peak_vram_gb:.2f} GB")

    with open(args.output, "w") as f:
        json.dump(
            {
                "scores": results,
                "batch_size": args.batch_size,
                "iter_step": args.iter_step,
                "load_time_sec": load_time,
                "load_peak_vram_gb": load_peak_gb,
                "infer_time_sec": infer_time,
                "infer_peak_vram_gb": peak_vram_gb,
            },
            f,
            indent=2,
        )


if __name__ == "__main__":
    main()
