"""Export a merged HPSv3 checkpoint as serialized bitsandbytes NF4."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))



def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--merged-dir", "--model", dest="model", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    from evaluation._conversion import export_bnb
    output = Path(args.output_dir)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Output directory must be empty")
    export_bnb(args.model, str(output))
    print(f"EXPORT_COMPLETE {output}")


if __name__ == "__main__":
    main()
