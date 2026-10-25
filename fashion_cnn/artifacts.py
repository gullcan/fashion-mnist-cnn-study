"""Small, dependency-free artifact IO and snapshot integrity checks."""

import hashlib
import json
import os
import tempfile
from pathlib import Path


def save_json(path, value):
    """Write metadata atomically; a serialization failure leaves the old file intact."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, encoding="utf-8",
                                         newline="\n", delete=False) as file:
            temporary = Path(file.name)
            json.dump(value, file, indent=2, ensure_ascii=False, allow_nan=False)
            file.write("\n")
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_manifest(root):
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
