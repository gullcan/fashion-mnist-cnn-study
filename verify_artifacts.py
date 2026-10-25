"""Verify the packaged snapshot without importing the ML stack or downloading data."""

import hashlib
import json
from pathlib import Path


def verify(root):
    root = Path(root)
    manifest = json.loads((root / "artifact_manifest.json").read_text(encoding="utf-8"))
    failures = []
    for record in manifest["files"]:
        path = root / record["path"]
        if not path.is_file():
            failures.append(f"Missing: {record['path']}")
        elif path.stat().st_size != record["bytes"] or hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]:
            failures.append(f"Changed: {record['path']}")
    if failures:
        raise RuntimeError("\n".join(failures))
    print(f"Verified {len(manifest['files'])} packaged files. No training or downloads performed.")


if __name__ == "__main__":
    verify(Path(__file__).resolve().parent)

