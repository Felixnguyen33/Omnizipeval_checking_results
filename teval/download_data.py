#!/usr/bin/env python3
"""Download the T-Eval English v2 data files at a pinned Hub revision and verify their sha256.

The files come from the Hugging Face dataset ``lovesnowbest/T-Eval`` (Apache-2.0), the
data release referenced by the upstream T-Eval README. They are written to ``teval/data/``,
which is where ``evaluate.py`` reads them. Existing files are re-verified, not re-downloaded.

    python teval/download_data.py
"""
import hashlib
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
DATA_REPO = "lovesnowbest/T-Eval"
DATA_REVISION = "af355ab2b62cdbae4262ac41c7529ffeae395012"  # Hub commit of 2024-02-18
DATA_SHA256 = {
    "instruct_v2.json": "01a53fa5dc1d648dde7b5c727fbb523048272c55b4acd6dd47e59ae5edf98272",
    "plan_json_v2.json": "58988cb147f434c8af0da4dc805de1124612138709213235c6dc5918b6334b2b",
    "plan_str_v2.json": "aa837d197969ee766c64cc3a2b7071e8776af53e00b0f36a3ccc9c878e865486",
    "reason_str_v2.json": "bb0a63733ae5cede595d3e66e1e21de1fdf52c5f8258c79171c2aedb5925517f",
    "retrieve_str_v2.json": "5c9489586b6d765c947df21e17585246dd5e9c2effbdd3fc4f57cab6847ad764",
    "understand_str_v2.json": "f825ed6dd19c1c146be3b8ade417fbbea529ed0ae7374d851637b85b3b5409c8",
    "review_str_v2.json": "753bb78f026e9f2d1d9fdfe4a1fee77106f9e2ee507a205b9fd3f39e5d6f41c2",
    "reason_retrieve_understand_json_v2.json": "ed6c3021d8bbb5fb8bf2396521af80a9d6ec91cff2a20661bc41e4e76a1d771e",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(data_dir: Path = DATA_DIR) -> dict:
    """Return {file: sha256}; raise SystemExit if any file is missing or differs from the pin."""
    problems, hashes = [], {}
    for name, expected in DATA_SHA256.items():
        path = data_dir / name
        if not path.exists():
            problems.append(f"{name}: missing")
            continue
        hashes[name] = sha256(path)
        if hashes[name] != expected:
            problems.append(f"{name}: sha256 {hashes[name]} != pinned {expected}")
    if problems:
        raise SystemExit("T-Eval data check failed (run `python teval/download_data.py`):\n  " + "\n  ".join(problems))
    return hashes


def main():
    from huggingface_hub import hf_hub_download
    DATA_DIR.mkdir(exist_ok=True)
    for name, expected in DATA_SHA256.items():
        target = DATA_DIR / name
        if target.exists() and sha256(target) == expected:
            print(f"ok      {name}")
            continue
        cached = Path(hf_hub_download(DATA_REPO, f"data/{name}", repo_type="dataset", revision=DATA_REVISION))
        if sha256(cached) != expected:
            sys.exit(f"{name}: downloaded file does not match the pinned sha256")
        shutil.copyfile(cached, target)
        print(f"fetched {name}")
    verify()
    print(f"All {len(DATA_SHA256)} files verified in {DATA_DIR}")


if __name__ == "__main__":
    main()
