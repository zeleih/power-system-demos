from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results" / "reference" / "artifact_manifest.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    paths = [
        ROOT / "cases" / "cepri36" / "data" / "andes" / "CEPRI36_andes.xlsx",
        ROOT / "cases" / "il200" / "data" / "IL200_ffr.xlsx",
        ROOT / "cases" / "cepri36" / "outputs" / "summary.json",
        ROOT / "cases" / "cepri36" / "outputs" / "andes" / "summary.json",
        ROOT / "cases" / "il200" / "outputs" / "summary.json",
        ROOT / "notebooks" / "analytical_gscr_walkthrough.ipynb",
    ]
    paths.extend(
        sorted((ROOT / "cases" / "cepri36" / "data" / "raw_psasp").glob("*"))
    )
    records = [
        {
            "path": path.relative_to(ROOT).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        for path in paths
        if path.is_file()
    ]
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(records, indent=2), encoding="utf-8")
    print(f"Wrote {len(records)} artifact hashes to {OUTPUT}")


if __name__ == "__main__":
    main()
