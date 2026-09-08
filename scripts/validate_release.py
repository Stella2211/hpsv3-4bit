"""Compare local or default Hub NF4 inference using synthetic public test inputs.

Run each phase in a fresh process in the corresponding uv project.
Reports are local artifacts and are not uploaded with the weights.
"""
import argparse
import json
import math
from pathlib import Path
import sys
import time

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("project", choices=["hpsv3", "hpsv3pp"])
parser.add_argument("--model")
parser.add_argument("--output", required=True)
parser.add_argument("--baseline")
parser.add_argument("--local-files-only", action="store_true")
args = parser.parse_args()
root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / args.project / "src"))
import torch
from PIL import Image, ImageDraw

if args.project == "hpsv3":
    from evaluation.hpsv3_quantized import HPSv3QuantizedInferencer as Inferencer
else:
    from evaluation.hpsv3pp_quantized import HPSv3PPQuantizedInferencer as Inferencer

images = [Image.new("RGB", (448, 448), "white"), Image.new("RGB", (448, 448), "navy")]
ImageDraw.Draw(images[0]).rectangle((112, 112, 336, 336), fill="red")
ImageDraw.Draw(images[1]).ellipse((100, 100, 348, 348), fill="yellow")
prompts = ["A red square on a white background.", "A yellow circle on a dark blue background."]
start = time.perf_counter()
inf = Inferencer.from_merged_dir(args.model, local_files_only=args.local_files_only)
torch.cuda.synchronize()
load_seconds = time.perf_counter() - start
scores = inf.score(images, prompts)
assert all(math.isfinite(x) for x in scores)
captions = inf.caption(images[:1], max_new_tokens=16)
assert captions and captions[0].strip()
result = dict(scores=scores, captions=captions, load_seconds=load_seconds,
              peak_vram_gb=torch.cuda.max_memory_allocated() / 1e9)
if args.baseline:
    baseline = json.loads(Path(args.baseline).read_text())
    errors = [abs(a-b) for a,b in zip(scores, baseline["scores"], strict=True)]
    result["max_absolute_error"] = max(errors)
    result["caption_equal"] = captions == baseline["captions"]
    assert max(errors) <= 1e-5, errors
    assert result["caption_equal"], "Caption changed across serialization"
Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result))
