"""Score HPSv3 image/prompt pairs and print scores, timing and peak VRAM.

The published NF4 model downloads automatically on first use.
Usage (from the repository root):
    uv run python hpsv3/scripts/smoke_test.py --image img.png --prompt "a cat"
"""

import argparse
import time

import torch
from PIL import Image

from hpsv3_4bit.model_source import resolve_model_source
from hpsv3_4bit.runtime import load_model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", "--merged-dir", dest="merged_dir", help="Local checkpoint or Hub repo ID; defaults to the published bnb NF4 model")
    parser.add_argument("--revision", help="Hub branch, tag or commit (default: main)")
    parser.add_argument("--local-files-only", action="store_true", help="Use only cached/local model files")
    parser.add_argument(
        "--processor-dir",
        default=None,
        help="Where to load the tokenizer/processor from (default: selected model if it "
        "contains processor files, otherwise the Qwen/Qwen2-VL-7B-Instruct base model)",
    )
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
    # Initialize CUDA before querying memory statistics.
    torch.zeros(1, device="cuda:0")
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats(0)

    t0 = time.time()
    model_dir, processor_dir = resolve_model_source("hpsv3", args.merged_dir, revision=args.revision, local_files_only=args.local_files_only, processor_dir=args.processor_dir)
    session = load_model("hpsv3", model_dir, device="cuda:0", processor_directory=processor_dir)
    load_time = time.time() - t0
    load_peak_gb = torch.cuda.max_memory_allocated(0) / 1e9
    print(f"Loaded 4-bit model in {load_time:.1f}s, peak VRAM after load: {load_peak_gb:.2f} GB")

    torch.cuda.reset_peak_memory_stats(0)
    t0 = time.time()
    images = []
    for path in args.image:
        with Image.open(path) as image:
            images.append(image.convert("RGB"))
    scores = session.score_batch(images, args.prompt)
    infer_time = time.time() - t0
    infer_peak_gb = torch.cuda.max_memory_allocated(0) / 1e9

    for path, s in zip(args.image, scores):
        print(f"{path}: score={s:.4f}")
    print(f"Inference time: {infer_time:.2f}s, peak VRAM during inference: {infer_peak_gb:.2f} GB")
    overall_peak_gb = max(load_peak_gb, infer_peak_gb)
    print(f"OVERALL PEAK VRAM: {overall_peak_gb:.2f} GB")
