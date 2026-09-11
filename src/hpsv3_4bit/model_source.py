"""CLI-only resolution of model and processor snapshots.

The runtime deliberately accepts local directories only.  This module keeps
Hub access at the command-line boundary and returns paths that are safe to
pass to :func:`hpsv3_4bit.runtime.load_model`.
"""

from __future__ import annotations

from pathlib import Path

DEFAULT_MODEL_IDS = {
    "hpsv3": "stella221125/HPSv3-bnb-NF4",
    "hpsv3pp": "stella221125/HPSv3-PlusPlus-bnb-NF4",
}
BASE_PROCESSOR_IDS = {
    "hpsv3": "Qwen/Qwen2-VL-7B-Instruct",
    "hpsv3pp": "Qwen/Qwen3-VL-8B-Instruct",
}


def _local_or_repo(source: str | Path | None) -> tuple[Path | None, str | None]:
    if source is None:
        return None, None
    value = str(source)
    path = Path(value).expanduser()
    if path.is_dir():
        return path.resolve(), None
    # Match the legacy resolver: path-looking values are errors, while a bare
    # value is a Hub repository identifier.
    if value.startswith((".", "/", "\\")) or ":" in value:
        raise FileNotFoundError(f"Model directory does not exist: {source}")
    return None, value


PROCESSOR_PATTERNS = [
    "preprocessor_config.json", "processor_config.json", "tokenizer*", "vocab*",
    "merges.txt", "special_tokens_map.json", "chat_template*",
]


def _snapshot(repo_id: str, *, revision: str | None, local_files_only: bool,
              processor_only: bool = False) -> Path:
    from huggingface_hub import snapshot_download

    options = {"revision": revision, "local_files_only": local_files_only}
    if processor_only:
        options["allow_patterns"] = PROCESSOR_PATTERNS
    return Path(snapshot_download(repo_id, **options)).resolve()


def resolve_model_source(
    family: str,
    source: str | Path | None = None,
    *,
    revision: str | None = None,
    local_files_only: bool = False,
    processor_dir: str | Path | None = None,
) -> tuple[Path, Path]:
    """Resolve ``(model_snapshot, processor_snapshot)`` for a CLI run."""
    if family not in DEFAULT_MODEL_IDS:
        raise ValueError(f"Unknown HPS model family: {family}")
    local_model, repo_id = _local_or_repo(source)
    model_dir = local_model or _snapshot(repo_id or DEFAULT_MODEL_IDS[family], revision=revision,
                                         local_files_only=local_files_only)
    if processor_dir is not None:
        explicit_path, processor_repo = _local_or_repo(processor_dir)
        if explicit_path is not None:
            return model_dir, explicit_path
        # A processor Hub ID may be the selected model repo (and therefore
        # must use its revision); an independent base-model ID uses Hub's
        # default revision, matching the legacy loader.
        selected_repo = repo_id or DEFAULT_MODEL_IDS[family]
        processor_revision = revision if processor_repo == selected_repo else None
        return model_dir, _snapshot(processor_repo, revision=processor_revision,
                                    local_files_only=local_files_only, processor_only=True)
    # Merged exports containing their processor should use the matching files.
    if (model_dir / "preprocessor_config.json").is_file():
        return model_dir, model_dir
    processor = _snapshot(BASE_PROCESSOR_IDS[family], revision=None, local_files_only=local_files_only,
                          processor_only=True)
    return model_dir, processor
