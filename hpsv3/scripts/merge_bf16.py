"""Build a merged HPSv3 BF16 checkpoint on CPU for model conversion.

Downloads the base model and original reward checkpoint. Requires substantial
host RAM and disk space; unnecessary for scoring with the published NF4 model.

Usage (from the repository root):
    uv run --project hpsv3 hpsv3/scripts/merge_bf16.py --output-dir /path/to/merged-bf16
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
