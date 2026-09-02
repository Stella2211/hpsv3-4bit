"""One-time step: build HPSv3 in bf16 and save a merged full-precision
checkpoint to local disk, so it can later be re-loaded with a bitsandbytes
quantization_config (see src/evaluation/hpsv3_quantized.py for why this two
stage approach is necessary).

Runs on CPU only; no GPU is touched. Downloads Qwen2-VL-7B-Instruct and
HPSv3.safetensors from the Hugging Face Hub on first run (set HF_HOME to
control the cache location).

Usage (from the repository root):
    uv run --project hpsv3 hpsv3/scripts/merge_bf16.py \
        --output-dir /path/to/hpsv3-merged-bf16
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluation.hpsv3_quantized import merge_and_save_bf16

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory to write the merged bf16 model + processor to (~17GB)",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Path to a local HPSv3.safetensors (default: download from MizzenAI/HPSv3)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run the merge even if the output dir already has a MERGE_COMPLETE marker",
    )
    args = parser.parse_args()

    t0 = time.time()
    print("Starting HPSv3 bf16 merge...")
    out = merge_and_save_bf16(args.output_dir, checkpoint_path=args.checkpoint, force=args.force)
    print(f"Merged model saved to {out} in {time.time() - t0:.1f}s")
