#!/usr/bin/env python3
"""pass@1 for one LiveCodeBench code-generation run (OmniZipEval protocol).

pass@1 = mean over problems of (correct samples / n). With n=1 this is the
fraction of problems solved; with n=10 it is the mean success rate of the ten
samples, not "solved by any of ten". This equals the `Pass@1` line printed by
`python -m lcb_runner.evaluation.compute_scores --eval_all_file <file>`.

The run is rejected if the problem count or the per-problem sample count does
not match the protocol (release_v4 without a date filter = 713 problems).

Usage:
  python scripts/omnizipeval/score.py <run>/output/<repr>/Scenario.codegeneration_<n>_<T>_eval_all.json \
      --expected-problems 713 --n 10 [--out score.json]
"""
import argparse
import json
import sys


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("eval_all_file")
    ap.add_argument("--expected-problems", type=int, default=713)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    with open(args.eval_all_file) as f:
        results = json.load(f)

    errors = []
    ids = [r["question_id"] for r in results]
    if len(ids) != args.expected_problems:
        errors.append(f"{len(ids)} problems, expected {args.expected_problems}")
    if len(set(ids)) != len(ids):
        errors.append("duplicate question_id")
    bad_n = [r["question_id"] for r in results if len(r["graded_list"]) != args.n]
    if bad_n:
        errors.append(f"{len(bad_n)} problems without exactly n={args.n} graded samples")

    def rate(rows):
        if not rows:
            return None
        return 100.0 * sum(sum(r["graded_list"]) / len(r["graded_list"]) for r in rows) / len(rows)

    by_diff = {d: [r for r in results if r.get("difficulty") == d] for d in ("easy", "medium", "hard")}
    empty = sum(1 for r in results for c in r["code_list"] if not c.strip())
    dates = sorted(r["contest_date"] for r in results)
    score = {
        "eval_all_file": args.eval_all_file,
        "problems": len(results),
        "n": args.n,
        "contest_date_min": dates[0] if dates else None,
        "contest_date_max": dates[-1] if dates else None,
        "pass@1": rate(results),
        **{f"pass@1_{d}": rate(rows) for d, rows in by_diff.items()},
        **{f"problems_{d}": len(rows) for d, rows in by_diff.items()},
        "empty_extracted_programs": empty,
        "valid": not errors,
        "errors": errors,
    }
    print(json.dumps(score, indent=2))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(score, f, indent=2)
    if errors:
        print("INVALID RUN: " + "; ".join(errors), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
