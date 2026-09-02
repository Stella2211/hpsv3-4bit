"""One-time CPU-only merge: bf16 Qwen3-VL-8B skeleton + hpsv3++.pth checkpoint
-> saved full-precision model dir, ready for 4-bit reload. See
src/evaluation/hpsv3pp_quantized.py for why this two-stage approach is
needed (bitsandbytes shape mismatch if quantized before the checkpoint is
applied).

Deliberately does NOT touch CUDA_VISIBLE_DEVICES / the GPU at all -- this
step runs entirely on CPU. Needs ~40GB of host RAM.

Note: if you use the community pre-merged bf16 export
(bdsqlsz/HPSV3-PlusPLus-BF16 on the Hugging Face Hub), you can skip this
script entirely and point score_batch.py's --merged-dir at that download.

Usage (from the repository root):
    uv run --project hpsv3pp hpsv3pp/scripts/merge_bf16.py \
        --output-dir /path/to/hpsv3pp-merged-bf16
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
