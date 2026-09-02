"""Batch HPSv3++ scoring CLI, mirroring hpsv3/scripts/score_batch.py.

Reads a JSON list of {"id", "image", "prompt", ...} records, loads the
4-bit HPSv3++ model once, scores every record, and writes
{"id": ..., "score": ...} results plus timing/VRAM metadata as JSON.

Usage (from the repository root):
    CUDA_VISIBLE_DEVICES=0 uv run --project hpsv3pp hpsv3pp/scripts/score_batch.py \
        --merged-dir /path/to/hpsv3pp-merged-bf16 \
        --input records.json --output scores.json

Use CUDA_VISIBLE_DEVICES to pick the GPU. HPSv3++ in 4-bit needs ~7.1GB
peak VRAM.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Path to JSON list of {id, image, prompt} records")
    parser.add_argument("--output", required=True, help="Path to write JSON results")
    parser.add_argument("--merged-dir", required=True, help="Path to the merged bf16 HPSv3++ checkpoint")
    parser.add_argument("--batch-size", type=int, default=2, help="Images scored per forward pass (VRAM/time tradeoff)")
    parser.add_argument(
        "--iter-step",
        type=float,
        default=0.0,
        help="HPSv3++ conditioning value in [0, 1] (normalized RL-iteration condition); "
        "0.0 = plain preference scoring, as recommended upstream",
    )
    args = parser.parse_args()

    with open(args.input) as f:
        records = json.load(f)
    if not records:
        with open(args.output, "w") as f:
            json.dump({"scores": [], "load_time_sec": 0.0, "infer_time_sec": 0.0, "peak_vram_gb": 0.0}, f)
        return

    import torch
    from evaluation.hpsv3pp_quantized import HPSv3PPQuantizedInferencer

    assert torch.cuda.is_available(), "CUDA not visible -- check CUDA_VISIBLE_DEVICES"
    torch.zeros(1, device="cuda:0")
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats(0)

    t0 = time.time()
    inferencer = HPSv3PPQuantizedInferencer.from_merged_dir(args.merged_dir, device="cuda:0")
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
        prompts = [r["prompt"] for r in chunk]
        scores = inferencer.score(images, prompts, iter_step=args.iter_step)
        for r, s in zip(chunk, scores):
            results.append({"id": r["id"], "score": s})
        print(f"  scored {i + len(chunk)}/{len(records)}", flush=True)

    infer_time = time.time() - t0
    peak_vram_gb = torch.cuda.max_memory_allocated(0) / 1e9
    print(f"Scored {len(records)} records in {infer_time:.1f}s, peak VRAM: {peak_vram_gb:.2f} GB")

    with open(args.output, "w") as f:
        json.dump(
            {
                "scores": results,
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
