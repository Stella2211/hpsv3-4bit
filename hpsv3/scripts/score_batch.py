"""Batch HPSv3 scoring CLI.

Reads a JSON list of ``{"id": ..., "image": <path>, "prompt": <text>}``
records, loads the 4-bit HPSv3 model once, scores every record in batched
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Path to JSON list of {id, image, prompt} records")
    parser.add_argument("--output", required=True, help="Path to write JSON results")
    parser.add_argument("--merged-dir", required=True, help="Path to the merged bf16 HPSv3 checkpoint")
    parser.add_argument("--batch-size", type=int, default=4, help="Images scored per forward pass (VRAM/time tradeoff)")
    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        records = json.load(f)
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
    inferencer = HPSv3QuantizedInferencer.from_merged_dir(args.merged_dir, device="cuda:0")
    load_time = time.time() - t0
    load_peak_gb = torch.cuda.max_memory_allocated(0) / 1e9
    print(f"Loaded 4-bit HPSv3 in {load_time:.1f}s, peak VRAM after load: {load_peak_gb:.2f} GB")

    torch.cuda.reset_peak_memory_stats(0)
    t0 = time.time()

    images = [r["image"] for r in records]
    prompts = [r["prompt"] for r in records]
    scores: list[float] = []
    for i in range(0, len(images), args.batch_size):
        chunk_images = images[i : i + args.batch_size]
        chunk_prompts = prompts[i : i + args.batch_size]
        scores.extend(inferencer.score(chunk_images, chunk_prompts))

    infer_time = time.time() - t0
    infer_peak_gb = torch.cuda.max_memory_allocated(0) / 1e9
    overall_peak_gb = max(load_peak_gb, infer_peak_gb)
    print(f"Scored {len(records)} images in {infer_time:.2f}s, peak VRAM during inference: {infer_peak_gb:.2f} GB")

    result = {
        "scores": [{"id": r["id"], "score": s} for r, s in zip(records, scores)],
        "load_time_sec": load_time,
        "infer_time_sec": infer_time,
        "peak_vram_gb": overall_peak_gb,
    }
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
