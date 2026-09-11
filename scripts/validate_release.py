"""Validate a local or published NF4 release through the canonical runtime."""
import argparse
import json
import math
from pathlib import Path
import time

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("project", choices=["hpsv3", "hpsv3pp"])
parser.add_argument("--model")
parser.add_argument("--output", required=True)
parser.add_argument("--baseline")
parser.add_argument("--local-files-only", action="store_true")
parser.add_argument("--revision")
parser.add_argument("--processor-dir")
args = parser.parse_args()
from hpsv3_4bit.model_source import resolve_model_source
from hpsv3_4bit.runtime import load_model
import torch
from PIL import Image, ImageDraw

images = [Image.new("RGB", (448, 448), "white"), Image.new("RGB", (448, 448), "navy")]
ImageDraw.Draw(images[0]).rectangle((112, 112, 336, 336), fill="red")
ImageDraw.Draw(images[1]).ellipse((100, 100, 348, 348), fill="yellow")
prompts = ["A red square on a white background.", "A yellow circle on a dark blue background."]
model_dir, processor_dir = resolve_model_source(
    args.project, args.model, revision=args.revision,
    local_files_only=args.local_files_only, processor_dir=args.processor_dir,
)
start = time.perf_counter()
inf = load_model(args.project, model_dir, processor_directory=processor_dir)
torch.cuda.synchronize()
load_seconds = time.perf_counter() - start
iter_step = 0.0
scores = inf.score_batch(images, prompts, iter_step=iter_step)
assert all(math.isfinite(x) for x in scores)
caption = inf.caption(images[0], max_new_tokens=16)
assert caption.strip()
result = dict(scores=scores, captions=[caption], load_seconds=load_seconds,
              peak_vram_gb=torch.cuda.max_memory_allocated() / 1e9)
if args.baseline:
    baseline = json.loads(Path(args.baseline).read_text())
    errors = [abs(a-b) for a,b in zip(scores, baseline["scores"], strict=True)]
    result["max_absolute_error"] = max(errors)
    result["caption_equal"] = [caption] == baseline["captions"]
    assert max(errors) <= 1e-5, errors
    assert result["caption_equal"], "Caption changed across serialization"
Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result))
