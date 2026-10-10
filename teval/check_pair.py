#!/usr/bin/env python3
"""Check that a dense T-Eval run and a compressed-variant run are comparable, and re-verify both.

    python teval/check_pair.py /path/to/dense_run /path/to/variant_run

For the pair: ``fairness.compare_manifests`` (same protocol fingerprint, same items, same dense parent,
baseline is the parent, both checkpoint checks passed), and identical prompt token ids for every item.

For each run, from the saved files alone:
- the data files match the pinned sha256 and the run covers exactly the expected items, in order;
- the prompts are rebuilt from the data with the parent tokenizer and must hash to the saved prompt hashes;
- every prediction is re-extracted from the raw generation, and predictions/<subset>.json must equal the
  pinned data item plus that prediction;
- the upstream T-Eval evaluators re-score those predictions; every per-subset metric and the seven reported
  numbers must equal the saved ones, and the vendored upstream convert_results.py must print the same numbers.

Exit code 0 means the pair is fair and both runs re-verified.
"""
import argparse
import importlib.util
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "common"))
import fairness  # noqa: E402
import scoring  # noqa: E402
from download_data import DATA_SHA256, verify as verify_data  # noqa: E402

TOLERANCE = 1e-6  # on 0-1 metrics; CPU embedding scores are deterministic to float precision


def _runner():
    spec = importlib.util.spec_from_file_location("teval_runner", ROOT / "evaluate.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _parent_args(parent: dict):
    return (parent["path"], None) if "path" in parent else (parent["repo"], parent["revision"])


def verify_run(folder: Path) -> tuple[dict, dict, dict]:
    """Return (manifest, {item id: prompt sha256}, report) for one run folder."""
    manifest = fairness.load_manifest(folder)
    problems = []
    if manifest["benchmark"] != "teval":
        raise SystemExit(f"{folder}: not a T-Eval run")
    protocol = manifest["protocol"]
    if protocol["dataset"]["files_sha256"] != DATA_SHA256:
        problems.append("run used data files that differ from the pinned T-Eval v2 files")
    verify_data()

    items = scoring.load_items(protocol.get("smoke"))
    ids = [i["id"] for i in items]
    if json.loads((folder / "items.json").read_text()) != sorted(ids):
        problems.append("items.json does not list exactly the protocol's items")

    # Rebuild prompts from the pinned data and the parent tokenizer.
    from transformers import AutoTokenizer
    parent, parent_revision = _parent_args(manifest["parent"])
    tokenizer = AutoTokenizer.from_pretrained(parent, revision=parent_revision, trust_remote_code=False)
    _runner().build_prompts(tokenizer, items)

    rows = [json.loads(line) for line in (folder / "generations.jsonl").open()]
    if [r["id"] for r in rows] != ids:
        problems.append("generations.jsonl is incomplete, duplicated or out of order")
    predictions = {s["key"]: json.loads((folder / "predictions" / f"{s['key']}.json").read_text())
                   for s in scoring.SUBSETS}
    rebuilt = {s["key"]: {} for s in scoring.SUBSETS}
    mismatched_prompts = mismatched_predictions = mismatched_rows = 0
    for item, row in zip(items, rows):
        mismatched_prompts += (row["prompt_sha256"] != item["prompt_sha256"] or
                               row["opens_think"] != item["opens_think"])
        prediction, closed = scoring.extract_prediction(row["raw"], item["opens_think"])
        mismatched_predictions += (prediction != row["prediction"] or closed != row["reasoning_closed"])
        rebuilt[item["subset"]][item["key"]] = {**item["datum"], "prediction": prediction}
        mismatched_rows += predictions[item["subset"]].get(item["key"]) != rebuilt[item["subset"]][item["key"]]
    for name, count in (("prompt hashes", mismatched_prompts), ("re-extracted predictions", mismatched_predictions),
                        ("predictions/*.json entries", mismatched_rows)):
        if count:
            problems.append(f"{count} {name} differ from the run folder")
    if any(len(predictions[k]) != len(rebuilt[k]) for k in rebuilt):
        problems.append("predictions/*.json contain extra items")

    # Re-score the re-extracted predictions with the upstream evaluators.
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for key, entries in rebuilt.items():
            (tmp / f"{key}.json").write_text(json.dumps(entries, ensure_ascii=False))
        results = scoring.score_folder(tmp)
        (tmp / "results.json").write_text(json.dumps(results))
        upstream = scoring.upstream_convert_results(tmp / "results.json")
    saved_results = json.loads((folder / "scores_by_subset.json").read_text())
    metric_diffs = sorted(f"{k}.{m}" for k in results for m in set(results[k]) | set(saved_results.get(k, {}))
                          if not scoring.close(results[k].get(m), saved_results.get(k, {}).get(m), TOLERANCE))
    if metric_diffs:
        problems.append("re-scored metrics differ: " + ", ".join(metric_diffs))
    rescored = scoring.capabilities(results)
    score_diffs = [k for k in scoring.CAPABILITIES
                   if not scoring.close(rescored[k], manifest["scores"].get(k), TOLERANCE * 100)]
    if score_diffs:
        problems.append("re-scored capabilities differ from manifest scores: " + ", ".join(score_diffs))
    upstream_diffs = [k for k, u in zip(scoring.CAPABILITIES, upstream)
                      if not scoring.close(round(rescored[k], 1), u, 0.051)]
    if upstream_diffs:
        problems.append("upstream convert_results.py disagrees on: " + ", ".join(upstream_diffs))

    report = {"folder": str(folder), "verified": not problems, "problems": problems, "n_items": len(rows),
              "smoke": protocol.get("smoke"), "saved_scores": manifest["scores"], "rescored": rescored,
              "upstream_convert_results": dict(zip(scoring.CAPABILITIES, upstream)),
              "generation_stats": manifest.get("generation_stats"), "hardware": manifest.get("hardware")}
    return manifest, {r["id"]: r["prompt_sha256"] for r in rows}, report


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("baseline", type=Path, help="Dense parent run folder")
    p.add_argument("variant", type=Path, help="Compressed variant run folder")
    args = p.parse_args()
    base_manifest, base_prompts, base_report = verify_run(args.baseline)
    var_manifest, var_prompts, var_report = verify_run(args.variant)
    pair = fairness.compare_manifests(base_manifest, var_manifest)
    prompts_identical = base_prompts == var_prompts
    if not prompts_identical:
        pair["problems"].append("prompt token ids differ between the two runs")
    hw_keys = ("tensor_parallel_size", "gpus")
    hardware_differences = [k for k in hw_keys if (base_manifest.get("hardware") or {}).get(k) !=
                            (var_manifest.get("hardware") or {}).get(k)]
    ok = pair["fair"] and prompts_identical and base_report["verified"] and var_report["verified"]
    print(json.dumps({"ok": ok, "pair": pair, "prompts_identical": prompts_identical,
                      "hardware_differences": hardware_differences,
                      "baseline": base_report, "variant": var_report}, indent=2, default=str))
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
