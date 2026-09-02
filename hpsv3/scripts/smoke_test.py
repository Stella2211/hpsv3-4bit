"""Load HPSv3 in 4-bit and score one or more (image, prompt) pairs, printing
scores plus load/inference time and peak VRAM.

Usage (from the repository root):
    CUDA_VISIBLE_DEVICES=0 uv run --project hpsv3 hpsv3/scripts/smoke_test.py \
        --merged-dir /path/to/hpsv3-merged-bf16 \
        --image img1.png --prompt "a cat" \
        --image img2.png --prompt "a dog"
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from src.evaluation.hpsv3_quantized import HPSv3QuantizedInferencer

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--merged-dir", required=True, help="Path to the merged bf16 HPSv3 checkpoint")
    parser.add_argument("--image", action="append", required=True, help="Image path (repeatable)")
    parser.add_argument(
        "--prompt",
        action="append",
        required=True,
        help="Prompt paired with the image given in the same position (repeatable)",
    )
    args = parser.parse_args()
    if len(args.image) != len(args.prompt):
        parser.error(f"--image and --prompt counts must match ({len(args.image)} vs {len(args.prompt)})")

    assert torch.cuda.is_available(), "CUDA not visible -- check CUDA_VISIBLE_DEVICES"
    # Force lazy CUDA context init before touching memory-stats APIs.
    # torch.cuda.reset_peak_memory_stats(0) has been observed to raise
    # "RuntimeError: Invalid device argument" intermittently when called
    # before any tensor has actually been placed on the device (seen under
    # CUDA_VISIBLE_DEVICES remapping); a trivial cuda allocation + sync
    # first makes it reliable.
    torch.zeros(1, device="cuda:0")
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats(0)

    t0 = time.time()
    inferencer = HPSv3QuantizedInferencer.from_merged_dir(args.merged_dir, device="cuda:0")
    load_time = time.time() - t0
    load_peak_gb = torch.cuda.max_memory_allocated(0) / 1e9
    print(f"Loaded 4-bit model in {load_time:.1f}s, peak VRAM after load: {load_peak_gb:.2f} GB")

    torch.cuda.reset_peak_memory_stats(0)
    t0 = time.time()
    scores = inferencer.score(args.image, args.prompt)
    infer_time = time.time() - t0
    infer_peak_gb = torch.cuda.max_memory_allocated(0) / 1e9

    for path, s in zip(args.image, scores):
        print(f"{path}: score={s:.4f}")
    print(f"Inference time: {infer_time:.2f}s, peak VRAM during inference: {infer_peak_gb:.2f} GB")
    overall_peak_gb = max(load_peak_gb, infer_peak_gb)
    print(f"OVERALL PEAK VRAM: {overall_peak_gb:.2f} GB")
