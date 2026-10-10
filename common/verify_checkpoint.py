#!/usr/bin/env python3
"""Check one compressed checkpoint before evaluating it: parent identity, declared bits, sampled sparsity.

Example (Qwen2.5-3B-Instruct, SparseGPT 50%):
    python common/verify_checkpoint.py --model <org>/<compressed-checkpoint> \
        --revision <sha> --parent Qwen/Qwen2.5-3B-Instruct --parent-revision <sha> --expect-sparsity 0.5
Exit code 0 means every check passed.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fairness  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True)
    p.add_argument("--revision")
    p.add_argument("--parent", required=True)
    p.add_argument("--parent-revision")
    p.add_argument("--expect-bits", type=int)
    p.add_argument("--expect-sparsity", type=float)
    args = p.parse_args()
    fairness.identity(args.model, args.revision)
    fairness.identity(args.parent, args.parent_revision)
    report = fairness.checkpoint_report(args.model, args.revision, args.parent, args.parent_revision,
                                        fairness.expect_from_args(args))
    print(json.dumps(report, indent=2, default=str))
    raise SystemExit(1 if report["failures"] else 0)


if __name__ == "__main__":
    main()
