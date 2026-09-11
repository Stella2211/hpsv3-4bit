"""Small stable API around the existing family-specific inferencers.

The ComfyUI adapter owns batching and lifecycle policy; this module owns model
construction and one-pair inference. No training package, subprocess, remote
code, or runtime installation is used here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from PIL import Image


class HPSv3Session:
    def __init__(self, family: str, inferencer, check_cancel: Callable[[], None] | None = None):
        self.family = family
        self.inferencer = inferencer
        self.model = inferencer.model
        self._check_cancel = check_cancel or (lambda: None)

    def score(self, image: Image.Image, prompt: str) -> float:
        self._check_cancel()
        values = self.inferencer.score([image], [prompt])
        self._check_cancel()
        if len(values) != 1:
            raise RuntimeError(f"Expected one score, got {len(values)}")
        return float(values[0])

    def caption(self, image: Image.Image, max_new_tokens: int = 96, stopping_criteria=None) -> str:
        self._check_cancel()
        values = self.inferencer.caption([image], max_new_tokens=max_new_tokens, stopping_criteria=stopping_criteria)
        self._check_cancel()
        if len(values) != 1 or not values[0].strip():
            raise ValueError("Model returned an empty caption.")
        return values[0].strip()


def load_model(
    family: str,
    directory: str | Path,
    device: str = "cuda",
    check_cancel: Callable[[], None] | None = None,
) -> HPSv3Session:
    """Load a local merged NF4 model and return a one-pair inference session."""
    if family == "hpsv3":
        from .hpsv3.quantized import HPSv3QuantizedInferencer

        inferencer = HPSv3QuantizedInferencer.from_merged_dir(
            merged_dir=str(Path(directory)), device=str(device), check_cancel=check_cancel
        )
    elif family == "hpsv3pp":
        from .hpsv3pp.quantized import HPSv3PPQuantizedInferencer

        inferencer = HPSv3PPQuantizedInferencer.from_merged_dir(
            merged_dir=str(Path(directory)), device=str(device), check_cancel=check_cancel
        )
    else:
        raise ValueError(f"Unknown HPS model family: {family}")
    return HPSv3Session(family, inferencer, check_cancel)
