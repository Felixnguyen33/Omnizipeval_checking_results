#!/usr/bin/env python3
"""Check that a dense GPQA run and a compressed-variant run (from ``evaluate.py``) are comparable.

Exit code 0 only when:

- ``common/fairness.compare_manifests`` finds nothing unfair (same protocol fingerprint, same
  questions, baseline is the dense parent, variant names the same parent, both checkpoints
  passed the parent-embedding / bits / sparsity checks);
- rescoring every response with the task's own extractors gives the per-question and aggregate
  scores written by ``evaluate.py``, for both filters, in both runs;
- both runs saw the same questions, choice orders and rendered prompts.

The paper metric is ``exact_match,strict-match``; flexible-extract is printed for reference.
"""
import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "common"))
import fairness  # noqa: E402

spec = importlib.util.spec_from_file_location("gpqa_report_utils", ROOT / "scripts/eval/tasks/gpqa_qwen25_report/utils.py")
task_utils = importlib.util.module_from_spec(spec)
spec.loader.exec_module(task_utils)
EXTRACT = {"strict-match": task_utils.extract_simple_evals, "flexible-extract": task_utils.extract_flexible}


def rescore(folder: Path, manifest: dict):
    """Rescore every saved response; return {split: {doc_id: (doc, prompt)}} and rescored percentages."""
    results = json.loads((folder / "results.json").read_text())
    observed, rescored = {}, {}
    for split, task in manifest["protocol"]["tasks"].items():
        rows = [json.loads(line) for line in (folder / f"samples_{task}.jsonl").open()]
        observed[split], rescored[split] = {}, {}
        for name, extract in EXTRACT.items():
            values = []
            for row in (r for r in rows if r["filter"] == name):
                pred = extract(row["resps"][0][0]) or "[invalid]"
                if pred != row["filtered_resps"][0]:
                    raise SystemExit(f"{folder}: {task} doc {row['doc_id']} {name}: saved {row['filtered_resps'][0]!r}, "
                                     f"rescored {pred!r}")
                value = float(pred == row["target"])
                if not math.isclose(value, row["exact_match"], abs_tol=1e-12):
                    raise SystemExit(f"{folder}: {task} doc {row['doc_id']} {name}: per-question score mismatch")
                values.append(value)
                observed[split][row["doc_id"]] = (row["doc"], row["arguments"])
            score = 100 * sum(values) / len(values)
            saved = 100 * results["results"][task][f"exact_match,{name}"]
            if not (math.isclose(score, saved, abs_tol=1e-9) and math.isclose(score, manifest["scores"][split][name],
                                                                                  abs_tol=1e-9)):
                raise SystemExit(f"{folder}: {task} {name}: rescored {score} != saved {saved}")
            rescored[split][name] = round(score, 2)
    return observed, rescored


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("baseline", type=Path)
    p.add_argument("variant", type=Path)
    args = p.parse_args()
    base, variant = fairness.load_manifest(args.baseline), fairness.load_manifest(args.variant)
    result = fairness.compare_manifests(base, variant)
    problems = list(result["problems"])
    base_docs, base_scores = rescore(args.baseline, base)
    variant_docs, variant_scores = rescore(args.variant, variant)
    if base_docs != variant_docs:
        problems.append("questions, choice order or rendered prompts differ between the runs")
    hardware_diff = sorted(k for k in set(base.get("hardware", {})) | set(variant.get("hardware", {}))
                           if base.get("hardware", {}).get(k) != variant.get("hardware", {}).get(k))
    print(json.dumps({
        "fair": not problems, "problems": problems, "paper_metric": base["protocol"]["paper_metric"],
        "family": base["protocol"]["family"], "max_gen_toks": base["protocol"]["max_gen_toks"],
        "thinking": base["protocol"]["thinking"],
        "baseline_percent": base_scores, "variant_percent": variant_scores,
        "hit_max_gen_toks": {"baseline": {s: base["scores"][s]["hit_max_gen_toks"] for s in base_scores},
                             "variant": {s: variant["scores"][s]["hit_max_gen_toks"] for s in variant_scores}},
        "hardware_differences (warning)": hardware_diff,
        "package_version_differences": result["package_version_differences"],
        "variant_quantization": result["variant_quantization"], "variant_sparsity": result["variant_sparsity"],
    }, indent=2, default=str))
    raise SystemExit(0 if not problems else 1)


if __name__ == "__main__":
    main()
