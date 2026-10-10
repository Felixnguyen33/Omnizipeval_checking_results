#!/usr/bin/env python3
"""Re-run only the judge stage on a finished run's saved responses and compare with its saved judgments.

Writes NEW_DIR/judgments.jsonl and NEW_DIR/rejudge.json (scores and per-item label agreement).
The original run folder is not modified. Judge settings are the constants in evaluate.py.

    python rejudge.py out/dense out/dense_rejudge --judge-path /path/to/Qwen3Guard-Gen-4B --gpus 0
"""
import argparse
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


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run", type=Path, help="finished evaluate.py output folder")
    p.add_argument("output", type=Path, help="new folder for the re-judged labels (refused if it exists)")
    p.add_argument("--judge-path", type=Path)
    p.add_argument("--gpus", help="CUDA device id for the judge (default: inherit)")
    p.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    args = p.parse_args()
    manifest = fairness.load_manifest(args.run)
    if manifest["protocol"]["judge"] != xs.JUDGE or manifest["protocol"]["labels"] != xs.LABEL_RULES:
        raise SystemExit("the run used different judge settings than this evaluate.py")
    if args.output.exists():
        raise SystemExit(f"{args.output} already exists")
    judge_path = xs.resolve_judge(args.judge_path)
    args.output.mkdir(parents=True)
    xs.run_stage("judge", {"responses": str(args.run / "responses.jsonl"), "judge_path": judge_path,
                           "gpu_memory_utilization": args.gpu_memory_utilization,
                           "out": str(args.output / "judgments.jsonl")},
                 args.gpus.split(",")[0] if args.gpus else None, args.output / "judge.log")
    responses = xs.read_jsonl(args.run / "responses.jsonl")
    old = {j["id"]: j for j in xs.read_jsonl(args.run / "judgments.jsonl")}
    new = xs.read_jsonl(args.output / "judgments.jsonl")
    differ = [j["id"] for j in new if (j["safety"], j["refusal"]) != (old[j["id"]]["safety"], old[j["id"]]["refusal"])]
    scores = xs.compute_scores(responses, new)
    report = {"run": str(args.run), "items": len(new), "label_disagreements": len(differ), "disagreeing_ids": differ,
              "scores": scores, "original_scores": manifest["scores"]}
    (args.output / "rejudge.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k not in ("scores", "original_scores")}, indent=2))
    raise SystemExit(0 if not differ else 1)


if __name__ == "__main__":
    main()
