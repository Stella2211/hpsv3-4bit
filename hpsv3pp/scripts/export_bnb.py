"""Export a merged HPSv3++ checkpoint as serialized bitsandbytes NF4."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from evaluation.hpsv3pp_quantized import HPSv3PPQuantizedInferencer


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--merged-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    output = Path(args.output_dir)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Output directory must be empty")
    inf = HPSv3PPQuantizedInferencer.from_merged_dir(args.merged_dir)
    inf.model.save_pretrained(output, safe_serialization=True, max_shard_size="2GB")
    inf.processor.save_pretrained(output)
    print(f"EXPORT_COMPLETE {output}")
