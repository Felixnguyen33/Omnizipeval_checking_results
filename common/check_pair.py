#!/usr/bin/env python3
"""Compare a dense baseline run and a compressed-variant run of the same benchmark.

Works for every runner in this repository that writes ``manifest.json`` and ``items.json``
through ``common/fairness.py``. Exit code 0 means the pair is comparable:

- same benchmark and identical protocol fingerprint (prompt, decoding, budget, dataset, scorer, judge),
- same evaluated items,
- the baseline is the dense parent, and the variant names the same parent,
- both checkpoints passed the parent-embedding check (and any expected bits/sparsity).

Benchmark folders may add their own ``check_pair.py`` that also rescores saved outputs.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fairness  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("baseline", type=Path)
    p.add_argument("variant", type=Path)
    args = p.parse_args()
    result = fairness.compare_manifests(fairness.load_manifest(args.baseline), fairness.load_manifest(args.variant))
    print(json.dumps(result, indent=2, default=str))
    raise SystemExit(0 if result["fair"] else 1)


if __name__ == "__main__":
    main()
