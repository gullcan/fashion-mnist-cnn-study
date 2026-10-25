"""Refresh a reviewed snapshot after staging files; never changes result artifacts."""

import hashlib
import json
import subprocess
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    tracked = subprocess.run(["git", "ls-files", "-z"], cwd=root,
                             check=True, capture_output=True).stdout
    records = []
    for name in sorted(tracked.decode("utf-8").split("\0")):
        if not name or name == "artifact_manifest.json":
            continue
        content = (root / name).read_bytes()
        records.append({"path": name, "bytes": len(content),
                        "sha256": hashlib.sha256(content).hexdigest()})
    manifest = {"description": "Reviewed source and completed-study snapshot; original result bytes preserved.",
                "files": records}
    (root / "artifact_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8", newline="\n",
    )
    print(f"Manifest updated for {len(records)} tracked files. Stage it after review.")


if __name__ == "__main__":
    main()
