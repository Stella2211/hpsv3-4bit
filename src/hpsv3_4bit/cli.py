"""Shared batch scoring command-line interface for HPSv3 model families."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from PIL import Image

from .model_source import resolve_model_source
from .runtime import load_model

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}


def positive_int(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError(f"must be a positive integer, got {value!r}")
    return number


def unit_interval(value: str) -> float:
    number = float(value)
    if not 0 <= number <= 1:
        raise argparse.ArgumentTypeError(f"must be in [0, 1], got {value!r}")
    return number


def prompt_ext(value: str) -> str:
    ext = value if value.startswith(".") else "." + value
    if ext.lower() in IMAGE_EXTS:
        raise argparse.ArgumentTypeError(f"must not be an image extension, got {value!r}")
    return ext


def load_records(input_path: str, prompt_ext: str = ".txt", no_prompt: bool = False) -> list[dict]:
    path = Path(input_path)
    if path.is_dir():
        images = sorted(p for p in path.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
        if not no_prompt:
            missing = [p.name for p in images if not p.with_suffix(prompt_ext).is_file()]
            if missing:
                raise SystemExit(f"error: no {prompt_ext} prompt file found for {len(missing)} image(s) in {path}: "
                                 + ", ".join(missing) + "\nAdd the missing prompt files, or pass --no-prompt to auto-generate captions.")
        return [{"id": p.name, "image": str(p), **({} if no_prompt else {"prompt": p.with_suffix(prompt_ext).read_text(encoding="utf-8").strip()})} for p in images]
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def build_parser(family: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=f"Score image/prompt pairs with {family}.")
    parser.add_argument("--input", required=True, help="JSON records or an image directory")
    parser.add_argument("--output", required=True, help="Path to write JSON results")
    parser.add_argument("--model", "--merged-dir", dest="merged_dir", help="Local checkpoint or Hub repo ID")
    parser.add_argument("--revision", help="Hub branch, tag, or commit")
    parser.add_argument("--local-files-only", action="store_true", help="Use only cached/local Hub files")
    parser.add_argument("--processor-dir", help="Local processor directory or Hub repo ID")
    parser.add_argument("--batch-size", type=positive_int, default=4 if family == "hpsv3" else 2,
                        help="Images scored per forward pass")
    parser.add_argument("--prompt-ext", type=prompt_ext, default=".txt",
                        help="Prompt file extension for directory input (default: .txt)")
    parser.add_argument("--no-prompt", action="store_true", help="Generate captions and score against them")
    if family == "hpsv3pp":
        parser.add_argument("--iter-step", type=unit_interval, default=0.0,
                            help="HPSv3++ conditioning value in [0, 1] (default: 0.0)")
    return parser


def run(family: str, args: argparse.Namespace) -> dict:
    records = load_records(args.input, args.prompt_ext, args.no_prompt)
    if family == "hpsv3pp":
        empty = {"scores": [], "load_time_sec": 0.0, "load_peak_vram_gb": 0.0,
                 "infer_time_sec": 0.0, "infer_peak_vram_gb": 0.0}
    else:
        empty = {"scores": [], "load_time_sec": 0.0, "infer_time_sec": 0.0, "peak_vram_gb": 0.0}
    if not records:
        return empty
    import torch
    if not torch.cuda.is_available():
        raise AssertionError("CUDA not visible -- check CUDA_VISIBLE_DEVICES")
    torch.zeros(1, device="cuda:0")
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats(0)
    started = time.time()
    model_dir, processor_dir = resolve_model_source(family, args.merged_dir, revision=args.revision,
                                                     local_files_only=args.local_files_only,
                                                     processor_dir=args.processor_dir)
    session = load_model(family, model_dir, device="cuda:0", processor_directory=processor_dir)
    load_time = time.time() - started
    load_peak = torch.cuda.max_memory_allocated(0) / 1e9
    print(f"Loaded 4-bit {family} in {load_time:.1f}s, peak VRAM after load: {load_peak:.2f} GB")
    torch.cuda.reset_peak_memory_stats(0)
    started = time.time()
    rows = []
    for offset in range(0, len(records), args.batch_size):
        chunk = records[offset:offset + args.batch_size]
        images = []
        for record in chunk:
            with Image.open(record["image"]) as opened:
                images.append(opened.convert("RGB"))
        # The public session caption API intentionally remains single-image;
        # score_batch is the operation that preserves actual batch forwards.
        prompts = [session.caption(image) for image in images] if args.no_prompt else [r["prompt"] for r in chunk]
        kwargs = {"iter_step": args.iter_step} if family == "hpsv3pp" else {}
        scores = session.score_batch(images, prompts, **kwargs)
        for record, prompt, score in zip(chunk, prompts, scores):
            row = {"id": record["id"], "score": float(score)}
            if args.no_prompt:
                row["generated_prompt"] = prompt
            rows.append(row)
    torch.cuda.synchronize()
    infer_time = time.time() - started
    infer_peak = torch.cuda.max_memory_allocated(0) / 1e9
    print(f"Scored {len(records)} images in {infer_time:.2f}s, peak VRAM during inference: {infer_peak:.2f} GB")
    result = {"scores": rows, "load_time_sec": load_time, "infer_time_sec": infer_time,
              "peak_vram_gb": max(load_peak, infer_peak)}
    if family == "hpsv3pp":
        result = {"scores": rows, "batch_size": args.batch_size, "iter_step": args.iter_step,
                  "load_time_sec": load_time, "load_peak_vram_gb": load_peak,
                  "infer_time_sec": infer_time, "infer_peak_vram_gb": infer_peak}
    return result


def main(family: str, argv: list[str] | None = None) -> None:
    args = build_parser(family).parse_args(argv)
    result = run(family, args)
    with Path(args.output).open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)


def main_hpsv3(argv: list[str] | None = None) -> None:
    main("hpsv3", argv)


def main_hpsv3pp(argv: list[str] | None = None) -> None:
    main("hpsv3pp", argv)
