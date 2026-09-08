"""Verify public Hub file hashes against the prepared local release manifest."""
import argparse
import hashlib
import json
from pathlib import Path

from huggingface_hub import HfApi

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("directory")
args = parser.parse_args()
directory = Path(args.directory)
manifest = json.loads((directory / "release_manifest.json").read_text())
api = HfApi(token=False)
info = api.model_info(manifest["repo_id"], files_metadata=True)
assert not info.private, "Release is not public"
files = {f.rfilename: f for f in info.siblings}
expected = set(manifest["sha256"]) | {"release_manifest.json"}
assert set(files) - {".gitattributes"} == expected, (
    f"Remote file inventory differs: missing={sorted(expected - set(files))}, "
    f"extra={sorted(set(files) - expected - {'.gitattributes'})}"
)
for name, digest in manifest["sha256"].items():
    metadata = files[name]
    if metadata.lfs is not None:
        assert metadata.lfs.sha256 == digest, f"Remote weight hash mismatch: {name}"
    else:
        # Small config/docs files use Git's blob SHA-1, not raw file SHA-1.
        data = (directory / name).read_bytes()
        expected_blob = hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
        assert metadata.blob_id == expected_blob, f"Remote asset mismatch: {name}"
data = (directory / "release_manifest.json").read_bytes()
assert files["release_manifest.json"].blob_id == hashlib.sha1(
    b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest(), "Manifest mismatch"
print(json.dumps(dict(repo=info.id, revision=info.sha, verified_files=len(expected), public=True)))
