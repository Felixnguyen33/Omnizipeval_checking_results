#!/usr/bin/env python3
"""GPQA runner for OmniZipEval: one protocol for every model of a cohort.

Protocol (constants below, not CLI options):

- tasks ``gpqa_diamond_qwen25_report`` (198 questions) and ``gpqa_main_qwen25_report`` (448),
  both run for every model
- 0-shot, chat template of the dense parent, no system prompt
- OpenAI simple-evals prompt ("Answer: $LETTER") with simple-evals' seeded choice order
  (``random.Random(0)``), defined in ``scripts/eval/tasks/gpqa_qwen25_report/``
- greedy decoding (temperature 0, top-p 1, one sample), bfloat16, lm-eval + vLLM, seed 0
- generation budget: one constant per dense-parent family (``FAMILIES``), shared by every
  variant of that parent; the family is derived from the parent checkpoint, never from a flag
- scores: ``exact_match,strict-match`` is the primary (paper) metric; ``flexible-extract`` is also reported

Before loading the model, ``common/fairness`` checks that ``--model`` was compressed from
``--parent`` (identical token embeddings) and any ``--expect-bits`` / ``--expect-sparsity``.
Run the dense parent first (``--model`` = ``--parent``), then each variant with the same
``--parent``, then ``check_pair.py``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "common"))
import fairness  # noqa: E402

TASK_DIR = ROOT / "scripts/eval/tasks/gpqa_qwen25_report"
SPLITS = {"diamond": 198, "main": 448}  # split -> number of questions
TASK_NAME = "gpqa_{}_qwen25_report"
DATASET = "Idavidrein/gpqa"
DATASET_REVISION = "83022cefff930aea54f654c0b282e74b9eeda5c6"  # gated; accept the terms on the Hub
PAPER_METRIC = "exact_match,strict-match"
FILTERS = ("strict-match", "flexible-extract")
SEED = 0
DTYPE = "bfloat16"

# Generation budget per dense-parent family. Every variant of a parent uses its parent's row.
# Non-reasoning chat models: 8,192 generated tokens. Reasoning models (DeepSeek-R1 distills,
# Qwen3 with thinking): 32,768 generated tokens.
FAMILIES = {
    "qwen2.5": {"max_gen_toks": 8192, "max_model_len": 16384, "thinking": False,
                "enable_thinking_arg": None},
    "deepseek-r1-distill": {"max_gen_toks": 32768, "max_model_len": 40960, "thinking": True,
                            "enable_thinking_arg": None},  # the R1 template always opens <think>
    "qwen3": {"max_gen_toks": 32768, "max_model_len": 40960, "thinking": True,
              "enable_thinking_arg": None},  # Qwen3 template default: thinking on
}


def chat_template(model: str, revision: str | None) -> str:
    for name in ("chat_template.jinja", "tokenizer_config.json"):
        path = fairness.local_file(model, revision, name)
        if path is None:
            continue
        if name.endswith(".jinja"):
            return path.read_text()
        template = json.loads(path.read_text()).get("chat_template") or ""
        if isinstance(template, list):  # named templates
            template = next((t["template"] for t in template if t.get("name") == "default"), "")
        if template:
            return template
    return ""


def parent_family(parent: str, parent_revision: str | None) -> str:
    """Classify the dense parent from its config and chat template (not from a flag)."""
    model_type = fairness.read_config(parent, parent_revision).get("model_type")
    template = chat_template(parent, parent_revision)
    if "<｜Assistant｜>" in template:
        return "deepseek-r1-distill"
    if model_type in ("qwen3", "qwen3_moe") and "enable_thinking" in template:
        return "qwen3"
    if model_type == "qwen2" and "<|im_start|>" in template:
        return "qwen2.5"
    raise SystemExit(f"{parent}: unknown parent family (model_type={model_type}); add a FAMILIES entry "
                     "with a budget shared by every variant of this parent")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def serializable(value):
    if hasattr(value, "item"):
        return value.item()
    return str(value)


def build_protocol(family: str, smoke: int | None) -> dict:
    budget = FAMILIES[family]
    protocol = {
        "id": "gpqa_qwen25_report_v1", "dataset": DATASET, "dataset_revision": DATASET_REVISION,
        "tasks": {split: TASK_NAME.format(split) for split in SPLITS}, "questions": dict(SPLITS),
        "backend": "lm-eval+vllm", "dtype": DTYPE, "seed": SEED, "num_fewshot": 0,
        "chat_template": "dense parent", "system_instruction": None,
        "prompt": "simple-evals 0-shot CoT, choice order random.Random(0)",
        "decoding": {"do_sample": False, "temperature": 0.0, "top_p": 1.0, "samples": 1},
        "family": family, "max_gen_toks": budget["max_gen_toks"], "max_model_len": budget["max_model_len"],
        "thinking": budget["thinking"], "think_end_token": None,
        "paper_metric": PAPER_METRIC, "filters": list(FILTERS), "cache": False,
        "source_sha256": {"evaluate.py": sha256(Path(__file__))},
        "task_sha256": {p.name: sha256(p) for p in sorted(TASK_DIR.glob("*.yaml"))} | {
            "utils.py": sha256(TASK_DIR / "utils.py")},
        "versions": fairness.package_versions("lm-eval", "vllm", "transformers", "torch", "datasets"),
    }
    if smoke:
        protocol["smoke"] = smoke
    return protocol


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    fairness.add_model_arguments(p)
    p.add_argument("--tensor-parallel-size", type=int, default=1, help="Hardware (recorded): vLLM tensor parallel")
    p.add_argument("--gpu-memory-utilization", type=float, default=0.90, help="Hardware (recorded)")
    p.add_argument("--smoke", type=int, help="Only the first N questions of each split (recorded in the protocol, "
                                             "so check_pair.py refuses to pair it with a full run)")
    args = p.parse_args()
    if args.smoke is not None and args.smoke < 1:
        p.error("--smoke must be positive")
    if args.output.exists():
        p.error(f"{args.output} already exists; choose a new output directory")

    family = parent_family(args.parent, args.parent_revision)
    protocol = build_protocol(family, args.smoke)
    manifest = fairness.manifest_from_args(args, "gpqa", protocol)  # refuses unfair checkpoints
    manifest.data["hardware"] = {"tensor_parallel_size": args.tensor_parallel_size,
                                 "gpu_memory_utilization": args.gpu_memory_utilization}
    manifest.write()

    from lm_eval import evaluator
    from lm_eval.models.vllm_causallms import VLLM
    from lm_eval.tasks._yaml_loader import load_yaml
    from transformers import AutoTokenizer

    tasks = []
    for split in SPLITS:
        task = load_yaml(TASK_DIR / f"gpqa_{split}_qwen25_report.yaml", resolve_func=True, recursive=True)
        task["dataset_kwargs"] = {"revision": DATASET_REVISION}
        task["generation_kwargs"] = {**task["generation_kwargs"], "max_gen_toks": protocol["max_gen_toks"]}
        tasks.append(task)

    budget = FAMILIES[family]
    model = VLLM(pretrained=args.model, revision=args.revision, tokenizer=args.parent,
                 tokenizer_revision=args.parent_revision, dtype=DTYPE, seed=SEED, trust_remote_code=False,
                 tensor_parallel_size=args.tensor_parallel_size, gpu_memory_utilization=args.gpu_memory_utilization,
                 max_model_len=budget["max_model_len"], max_gen_toks=budget["max_gen_toks"], batch_size="auto",
                 enable_thinking=budget["enable_thinking_arg"])
    result = evaluator.simple_evaluate(
        model=model, tasks=tasks, num_fewshot=0, limit=args.smoke, use_cache=None, cache_requests=False,
        apply_chat_template=True, system_instruction=None, fewshot_as_multiturn=False, log_samples=True,
        random_seed=SEED, numpy_random_seed=SEED, torch_random_seed=SEED, fewshot_random_seed=SEED)

    tokenizer = AutoTokenizer.from_pretrained(args.parent, revision=args.parent_revision)
    samples = result.pop("samples")
    scores, item_ids = {"paper_metric": PAPER_METRIC}, []
    for split, full_size in SPLITS.items():
        task = TASK_NAME.format(split)
        expected = min(args.smoke or full_size, full_size)
        rows = samples[task]
        doc_ids = {r["doc_id"] for r in rows}
        if len(doc_ids) != expected or len(rows) != expected * len(FILTERS):
            raise SystemExit(f"{task}: {len(doc_ids)} questions / {len(rows)} rows, expected {expected}")
        with (args.output / f"samples_{task}.jsonl").open("w") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False, default=serializable) + "\n")
        responses = {r["doc_id"]: r["resps"][0][0] for r in rows}
        lengths = [len(tokenizer(text, add_special_tokens=False)["input_ids"]) for text in responses.values()]
        metrics = result["results"][task]
        scores[split] = {f: 100 * metrics[f"exact_match,{f}"] for f in FILTERS} | {
            "n": expected, "hit_max_gen_toks": sum(n >= protocol["max_gen_toks"] - 1 for n in lengths),
            "unparsed_strict": sum(r["filtered_resps"][0] == "[invalid]" for r in rows
                                   if r.get("filter") == "strict-match")}
        item_ids += [f"{split}:{d}" for d in sorted(doc_ids)]
    (args.output / "results.json").write_text(json.dumps(result, indent=2, default=serializable) + "\n")
    manifest.complete(scores, item_ids)
    print(json.dumps(scores, indent=2))


if __name__ == "__main__":
    main()
