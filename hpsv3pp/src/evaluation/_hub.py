"""Resolve a local checkpoint or a versioned Hub snapshot."""
import json
from pathlib import Path

DEFAULT_MODEL_ID = "stella221125/HPSv3-PlusPlus-bnb-NF4"


def resolve_model(source=None, *, revision=None, local_files_only=False):
    if source is not None and Path(source).is_dir():
        return str(Path(source))
    if source is not None and (str(source).startswith((".", "/", "\\")) or ":" in str(source)):
        raise FileNotFoundError(f"Model directory does not exist: {source}")
    from huggingface_hub import snapshot_download
    repo_id = str(source) if source is not None else DEFAULT_MODEL_ID
    return snapshot_download(
        repo_id, revision=revision, local_files_only=local_files_only,
    )


def load_reward_settings(directory, defaults, tokenizer):
    path = Path(directory) / "reward_config.json"
    if not path.exists():
        return defaults
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("format_version") != 1:
        raise ValueError("Unsupported reward_config format version")
    settings = data["model_kwargs"]
    if set(settings) != set(defaults):
        raise ValueError("Reward model settings do not match this loader")
    token_id = tokenizer.convert_tokens_to_ids(data["reward_token"])
    if token_id != data["reward_token_id"] or settings["special_token_ids"] != [token_id]:
        raise ValueError("Reward token ID does not match the saved tokenizer")
    return settings


def saved_quantization_config(directory):
    raw = json.loads((Path(directory) / "config.json").read_text(encoding="utf-8"))
    return raw.get("quantization_config")
