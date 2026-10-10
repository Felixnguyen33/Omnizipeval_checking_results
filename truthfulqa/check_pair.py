#!/usr/bin/env python3
"""Rescore two checked_v1 output folders and compare their observed protocols.

Also fails when either run's checkpoint checks (parent embeddings, expected bits or sparsity)
failed or were not recorded, when the two runs name different dense parents, or when the
baseline is not the dense parent itself.
"""
import argparse
import json
import math
from pathlib import Path



def unwrap(value):
    while isinstance(value, list) and len(value) == 1:
        value = value[0]
    return value


def rescore(folder):
    manifest = json.loads((folder / "manifest.json").read_text())
    if manifest.get("status") != "complete":
        raise ValueError(f"{folder}: run is incomplete")
    if manifest["protocol"]["dataset"] != 'truthfulqa':
        raise ValueError("Expected truthfulqa outputs")
    result = json.loads((folder / "results.json").read_text())
    observations, scores = {}, {}
    expected_tasks = {"truthfulqa_mc1", "truthfulqa_mc2"}
    if set(result["results"]) != expected_tasks:
        raise ValueError("Unexpected or missing tasks")
    for task in sorted(expected_tasks):
        rows = [json.loads(line) for line in (folder / f"samples_{task}.jsonl").open()]
        expected = manifest["protocol"].get("smoke") or 817
        if len(rows) != expected or len({r["doc_id"] for r in rows}) != expected:
            raise ValueError(f"{task}: missing or duplicate questions")
        observed, values = {}, []
        for row in rows:
            question = row["doc"]["question"]
            if question in observed:
                raise ValueError("Duplicate question")
            observed[question] = (row["doc"], row["arguments"])
            logits = [float(unwrap(cell)[0]) for cell in row["resps"]]
            labels = row["doc"][task.removeprefix("truthfulqa_") + "_targets"]["labels"]
            if len(logits) != len(labels) or not all(math.isfinite(x) for x in logits):
                raise ValueError("Invalid candidate likelihoods")
            if task.endswith("mc1"):
                value = labels[max(range(len(logits)), key=logits.__getitem__)]
            else:
                weights = [math.exp(x - max(logits)) for x in logits]
                value = sum(w * label for w, label in zip(weights, labels)) / sum(weights)
            metric = "acc"
            if not math.isclose(value, row[metric], abs_tol=1e-10, rel_tol=0):
                raise ValueError(f"{task}: per-question score mismatch")
            values.append(value)
        score = sum(values) / expected
        if not math.isclose(score, result["results"][task][metric + ",none"], abs_tol=1e-10, rel_tol=0):
            raise ValueError(f"{task}: aggregate mismatch")
        observations[task], scores[task] = observed, score * 100
    return manifest, observations, scores


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("baseline", type=Path)
    p.add_argument("variant", type=Path)
    args = p.parse_args()
    base, base_docs, base_scores = rescore(args.baseline)
    variant, variant_docs, variant_scores = rescore(args.variant)
    keys = set(base["protocol"]) | set(variant["protocol"])
    differences = sorted(k for k in keys if base["protocol"].get(k) != variant["protocol"].get(k))
    matched = base_docs == variant_docs
    checkpoint_problems = []
    for name, m in (("baseline", base), ("variant", variant)):
        if "checkpoint_checks" not in m:
            checkpoint_problems.append(f"{name}: no checkpoint checks recorded in manifest.json")
        elif not m.get("fair_checkpoint"):
            checkpoint_problems.append(f"{name} checkpoint checks failed: {m['checkpoint_checks'].get('failures')}")
    if base.get("parent") != variant.get("parent"):
        checkpoint_problems.append("the two runs name different dense parents")
    if base.get("model") != base.get("parent"):
        checkpoint_problems.append("the baseline is not the dense parent itself")
    print(json.dumps({"saved_scores_verified": True, "baseline_percent": base_scores,
                      "variant_percent": variant_scores, "protocol_differences": differences,
                      "documents_and_prompts_match": matched,
                      "variant_quantization_config": variant.get("quantization_config"),
                      "checkpoint_problems": checkpoint_problems,
                      "variant_checkpoint_checks": {k: variant.get("checkpoint_checks", {}).get(k)
                                                    for k in ("quantization", "sparsity", "parent")},
                      "fairness_certified": False,
                      "remaining_checks": ["Inspect generation truncation and compression calibration settings"]},
                     indent=2, default=str))
    raise SystemExit(0 if matched and not differences and not checkpoint_problems else 1)


if __name__ == "__main__":
    main()
