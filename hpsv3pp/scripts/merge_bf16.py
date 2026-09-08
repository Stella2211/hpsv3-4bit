"""Build a merged HPSv3++ BF16 checkpoint on CPU for model conversion.

Downloads the base model and original reward checkpoint. Requires substantial
host RAM and disk space; unnecessary for scoring with the published NF4 model.

Usage (from the repository root):
    uv run --project hpsv3pp hpsv3pp/scripts/merge_bf16.py --output-dir /path/to/merged-bf16
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from evaluation.hpsv3pp_quantized import merge_and_save_bf16, BASE_MODEL_NAME

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
        help="Path to a local hpsv3++.pth (default: download from Junjun2333/HPSv3-PlusPlus)",
    )
    parser.add_argument(
        "--base-model",
        default=BASE_MODEL_NAME,
        help=f"Base model id or local path (default: {BASE_MODEL_NAME})",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run the merge even if the output dir already has a MERGE_COMPLETE marker",
    )
    args = parser.parse_args()

    out = merge_and_save_bf16(
        output_dir=args.output_dir,
        checkpoint_path=args.checkpoint,
        base_model_name=args.base_model,
        force=args.force,
    )
    print("MERGE_DONE", out)
