#!/usr/bin/env python3
"""Check that a dense XSTest run and a compressed-variant run are comparable, and re-derive their scores.

1. ``fairness.compare_manifests``: same benchmark, identical protocol fingerprint (dataset, smoke size,
   template, decoding, judge, label rules, source hash), same items, same dense parent, checkpoint checks passed.
2. For each run: responses.jsonl and judgments.jsonl cover exactly items.json, every saved label re-parses
   from the raw judge output, and the scores recomputed from the two files equal manifest.json.
3. Both runs used the same chat prompt for every item.

Exit code 0 means all checks passed. Hardware differences (GPUs, tensor parallelism) are reported, not failed.
"""
import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "common"))
import fairness  # noqa: E402

spec = importlib.util.spec_from_file_location("xstest_evaluate", ROOT / "evaluate.py")
xs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(xs)

HEADLINE = ("harmful_refusal_pct", "unsafe_compliance_pct", "harmful_compliance_pct", "over_refusal_pct",
            "safe_compliance_pct", "unparsed", "truncated")


def verify_run(folder: Path, manifest: dict) -> tuple[list[str], dict]:
    problems = []
    items = json.loads((folder / "items.json").read_text())
    responses = xs.read_jsonl(folder / "responses.jsonl")
    judgments = xs.read_jsonl(folder / "judgments.jsonl")
    for name, rows in (("responses", responses), ("judgments", judgments)):
        if sorted(str(r["id"]) for r in rows) != items:
            problems.append(f"{folder}: {name}.jsonl does not cover items.json exactly")
    if hashlib.sha256(json.dumps(items).encode()).hexdigest() != manifest.get("items_sha256"):
        problems.append(f"{folder}: items.json does not match manifest items_sha256")
    if fairness.protocol_sha256(manifest["protocol"]) != manifest["protocol_sha256"]:
        problems.append(f"{folder}: protocol_sha256 does not match the recorded protocol")
    for j in judgments:
        parsed = xs.parse_judge(j["judge_raw"])
        if (parsed["safety"], parsed["refusal"]) != (j["safety"], j["refusal"]):
            problems.append(f"{folder}: item {j['id']} saved labels differ from re-parsed judge output")
            break
    try:
        rescored = xs.compute_scores(responses, judgments)
    except SystemExit as e:
        return problems + [f"{folder}: {e}"], {}
    if json.loads(json.dumps(rescored)) != manifest["scores"]:
        problems.append(f"{folder}: scores recomputed from saved judgments differ from manifest.json")
    return problems, {r["id"]: r["chat_prompt"] for r in responses}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("baseline", type=Path, help="output folder of the dense parent run")
    p.add_argument("variant", type=Path, help="output folder of the compressed run")
    args = p.parse_args()
    base, var = fairness.load_manifest(args.baseline), fairness.load_manifest(args.variant)
    result = fairness.compare_manifests(base, var)
    if base["benchmark"] != xs.BENCHMARK:
        result["problems"].append(f"not an {xs.BENCHMARK} run")
    if base["protocol"].get("source_sha256") != xs.build_protocol(None)["source_sha256"]:
        result["problems"].append("runs were produced by a different evaluate.py than the one in this folder")
    base_problems, base_prompts = verify_run(args.baseline, base)
    var_problems, var_prompts = verify_run(args.variant, var)
    result["problems"] += base_problems + var_problems
    if base_prompts and var_prompts and base_prompts != var_prompts:
        result["problems"].append("chat prompts differ between the two runs")
    result["fair"] = not result["problems"]
    result["scores_rederived_from_judgments"] = not (base_problems or var_problems)
    rb, rv = base.get("runtime", {}), var.get("runtime", {})
    result["runtime_differences"] = sorted(k for k in set(rb) | set(rv) if rb.get(k) != rv.get(k))
    result["headline"] = {k: {"baseline": base["scores"].get(k), "variant": var["scores"].get(k)} for k in HEADLINE}
    result["smoke"] = base["protocol"].get("smoke")
    result.pop("baseline_scores", None)
    result.pop("variant_scores", None)
    print(json.dumps(result, indent=2, default=str))
    raise SystemExit(0 if result["fair"] else 1)


if __name__ == "__main__":
    main()
