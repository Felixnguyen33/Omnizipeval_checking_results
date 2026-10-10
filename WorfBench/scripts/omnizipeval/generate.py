#!/usr/bin/env python3
"""WorfBench generation + node scoring for local models with vLLM (OmniZipEval protocol).

Upstream WorfBench (node_eval.py --task gen_workflow) queries an OpenAI-style
server one example at a time. This runner does the same prompting offline with
vLLM's chat API and then calls the unchanged upstream scorer:

  prompt   node_eval.build_message(conversations, few_shot, task)  (upstream)
           few-shot = the bundled two-shot examples where they exist
           (alfworld, lumos, os, toolalpaca, toolbench, webshop); intercodesql
           and seal_tools have none, so they run zero-shot.
  generate vllm.LLM.chat with the tokenizer's own chat template
           (--tokenizer, default: the model's own tokenizer).
  score    node_eval.eval_workflow(..., eval_type="node")  (upstream), with
           node_eval.NODE_PATTERN set by --node-parser.

Outputs in --output:
  settings.json                 every setting of the run
  pred/<task>.json              upstream prediction format ({"query", "workflow"})
                                plus finish_reason and n_tokens
  eval_<parser>/<task>_node.json   upstream score file (precision, recall, f1_score)
  summary_<parser>.json         per-task scores and the equal-weight 8-task mean

Prediction files resume when they are shorter than the gold file; finished
scores are reused. --score-only scores the pred/<task>.json of a finished
run (e.g. with --node-parser strict_space) without loading the generator.
"""

import argparse
import gc
import json
import os
import re
import sys
import time
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # the WorfBench checkout
sys.path.insert(0, str(ROOT))

import node_eval  # noqa: E402  (upstream module, minimal NODE_PATTERN change)
from prompts.eval_prompt import two_shot_example  # noqa: E402

TASKS = ["alfworld", "intercodesql", "lumos", "os", "seal_tools", "toolalpaca", "toolbench", "webshop"]
FEW_SHOT_TASKS = set(two_shot_example)
GOLD_ROOT = ROOT / "gold_traj"
NODE_PARSERS = {
    # OmniZipEval parser (default): "1:text", "1: text", "1.text", "1. text"
    "optional_space": r"\d+[:.]\s*(.+)",
    # upstream WorfBench parser: a space is required after "1:" / "1."
    "strict_space": r"\d+[:.] (.+)",
}
SCORER = "sentence-transformers/all-mpnet-base-v2"
SCORER_REVISION = "e8c3b32edf5434bc2275fc9bab85f82640a19130"


def load_json(path):
    with open(path) as f:
        return json.load(f)


def save_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")  # shards may write settings.json at once
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def gold_for(task, limit):
    gold = load_json(GOLD_ROOT / task / "graph_eval.json")
    return gold[:limit] if limit else gold


def shard_bounds(n, shard_id, num_shards):
    base, extra = divmod(n, num_shards)
    start = shard_id * base + min(shard_id, extra)
    return start, start + base + (1 if shard_id < extra else 0)


def make_llm(args):
    from vllm import LLM

    common = dict(
        model=args.model,
        tokenizer=args.tokenizer or args.model,
        trust_remote_code=True,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.90,
        dtype="auto",
        seed=args.seed,
    )
    try:
        return LLM(max_model_len=args.max_model_len, **common), args.max_model_len
    except Exception as exc:
        if not args.fallback_max_model_len:
            raise
        print(f"retry with max_model_len={args.fallback_max_model_len} after: {exc}", flush=True)
        return (LLM(max_model_len=args.fallback_max_model_len, enforce_eager=True, **common),
                args.fallback_max_model_len)


def release_llm(llm):
    import torch

    engine = getattr(llm, "llm_engine", None)
    core = getattr(engine, "engine_core", None) if engine is not None else None
    if core is not None and hasattr(core, "shutdown"):
        core.shutdown()
    del llm
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def generate(args, pred_path_for):
    from vllm import SamplingParams

    llm = None
    sampling = SamplingParams(temperature=args.temperature, top_p=args.top_p,
                              max_tokens=args.max_new_tokens, seed=args.seed)
    for task in args.tasks:
        gold = gold_for(task, args.limit)
        start, end = shard_bounds(len(gold), args.shard_id, args.num_shards)
        gold = gold[start:end]
        pred_path = pred_path_for(task)
        done = load_json(pred_path) if pred_path.exists() else []
        if len(done) > len(gold):
            raise SystemExit(f"{pred_path} has more rows than the gold file")
        pending = gold[len(done):]
        if not pending:
            print(f"skip gen {task}: {len(done)} rows complete", flush=True)
            continue
        if llm is None:
            llm, used_len = make_llm(args)
            print(f"vLLM max_model_len={used_len}", flush=True)
            save_json(args.output / f"runtime_shard{args.shard_id}.json", {"max_model_len": used_len})
        print(f"gen {task}: {len(done)} / {len(gold)} done", flush=True)
        batch = [node_eval.build_message(deepcopy(ex["conversations"]), task in FEW_SHOT_TASKS, task)
                 for ex in pending]
        t0 = time.time()
        outputs = llm.chat(batch, sampling, use_tqdm=True)
        for ex, out in zip(pending, outputs):
            o = out.outputs[0] if out.outputs else None
            done.append({"query": ex, "workflow": o.text if o else "",
                         "finish_reason": o.finish_reason if o else None,
                         "n_tokens": len(o.token_ids) if o else 0})
        save_json(pred_path, done)
        capped = sum(r.get("finish_reason") == "length" for r in done)
        print(f"saved {task} n={len(done)} length_capped={capped} in {time.time() - t0:.0f}s", flush=True)
    if llm is not None:
        release_llm(llm)


def merge_shards(args):
    for task in args.tasks:
        gold = gold_for(task, args.limit)
        merged = []
        for shard_id in range(args.merge_shards):
            part = args.output / "pred" / f"{task}.shard{shard_id}.json"
            rows = load_json(part)
            start, end = shard_bounds(len(gold), shard_id, args.merge_shards)
            if len(rows) != end - start:
                raise SystemExit(f"{part} has {len(rows)} rows, expected {end - start}")
            merged.extend(rows)
        save_json(args.output / "pred" / f"{task}.json", merged)
        print(f"merged {task} n={len(merged)}", flush=True)


def install_scorer(args):
    """Upstream scorer model at a pinned revision, on args.scorer_device.

    Embeddings are cached per string (upstream re-encodes the same node for each
    of up to 20 topological sorts). A workflow with no parsed nodes scores 0, as
    upstream intends, instead of reaching the encoder with an empty list.
    """
    import sentence_transformers
    import torch

    base = sentence_transformers.SentenceTransformer

    class CachedEncoder(base):
        def __init__(self, *a, **k):
            k["device"] = args.scorer_device
            k.setdefault("revision", args.scorer_revision)
            super().__init__(*a, **k)
            self._cache = {}

        def encode(self, sentences, **kwargs):
            keys = [sentences] if isinstance(sentences, str) else list(sentences)
            if not keys:
                return torch.empty((0, self.get_sentence_embedding_dimension()), device=self.device)
            missing = [t for t in dict.fromkeys(keys) if t not in self._cache]
            if missing:
                for text, vec in zip(missing, super().encode(missing, convert_to_tensor=True)):
                    self._cache[text] = vec.detach()
            return torch.stack([self._cache[t] for t in keys])

    sentence_transformers.SentenceTransformer = CachedEncoder
    upstream_eval_nodes = node_eval.t_eval_nodes

    def eval_nodes(pred_graph, gt_graph, sentence_model):
        pred = [n for n in pred_graph.get("nodes", []) if n not in ("START", "END")]
        gold = [n for n in gt_graph.get("nodes", []) if n not in ("START", "END")]
        if not pred or not gold:
            return {"precision": 0.0, "recall": 0.0, "f1_score": 0.0}
        return upstream_eval_nodes(pred_graph, gt_graph, sentence_model)

    node_eval.t_eval_nodes = eval_nodes


def score(args, settings):
    node_eval.NODE_PATTERN = re.compile(NODE_PARSERS[args.node_parser])
    install_scorer(args)
    eval_dir = args.output / f"eval_{args.node_parser}"
    eval_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for task in args.tasks:
        pred_path = args.output / "pred" / f"{task}.json"
        eval_path = eval_dir / f"{task}_node.json"
        gold = gold_for(task, args.limit)
        gold_path = GOLD_ROOT / task / "graph_eval.json"
        if args.limit:  # upstream eval_workflow reads the gold file, so write the truncated one
            gold_path = args.output / "gold_limited" / f"{task}.json"
            save_json(gold_path, gold)
        if not eval_path.exists():
            node_eval.eval_workflow(str(gold_path), str(pred_path), args.scorer, "node", str(eval_path))
        metrics = load_json(eval_path)
        rows.append({"task": task, **metrics, "n": len(gold), "few_shot": task in FEW_SHOT_TASKS})
        print(f"{task}: {metrics}", flush=True)
    avg = {k: sum(r[k] for r in rows) / len(rows) for k in ("precision", "recall", "f1_score")}
    summary = {**settings, "node_parser": args.node_parser, "tasks": rows, "average": avg,
               "average_x100": {k: round(100 * v, 2) for k, v in avg.items()}}
    save_json(args.output / f"summary_{args.node_parser}.json", summary)
    print("AVERAGE (node, equal task weights, x100):", json.dumps(summary["average_x100"]), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", help="HF id or local path (dense or compressed); not needed with --score-only")
    p.add_argument("--tokenizer", default=None,
                   help="Tokenizer + chat template (default: the model's own)")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--tasks", nargs="+", default=TASKS, choices=TASKS + ["wikihow"])
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--max-new-tokens", type=int, default=1024)
    p.add_argument("--max-model-len", type=int, default=16384)
    p.add_argument("--fallback-max-model-len", type=int, default=8192,
                   help="Retry length if the first vLLM load fails; 0 disables the retry")
    p.add_argument("--seed", type=int, default=0, help="vLLM engine/sampling seed (vLLM default 0)")
    p.add_argument("--node-parser", choices=sorted(NODE_PARSERS), default="optional_space")
    p.add_argument("--scorer", default=SCORER)
    p.add_argument("--scorer-revision", default=SCORER_REVISION)
    p.add_argument("--scorer-device", default="cuda")
    p.add_argument("--limit", type=int, default=0, help="Smoke test: first N examples per task (0 = all)")
    p.add_argument("--num-shards", type=int, default=1, help="Data-parallel generation shards")
    p.add_argument("--shard-id", type=int, default=0)
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--generate-only", action="store_true")
    mode.add_argument("--score-only", action="store_true", help="Score the existing pred/<task>.json of a finished run")
    mode.add_argument("--merge-shards", type=int, default=0, help="Merge N shard files, then score")
    args = p.parse_args()
    if not args.model and not args.score_only:
        p.error("--model is required unless --score-only")
    if not 0 <= args.shard_id < args.num_shards:
        p.error("need 0 <= --shard-id < --num-shards")
    if args.num_shards > 1 and not args.generate_only:
        p.error("sharded generation needs --generate-only; score with --merge-shards N")

    settings = {
        "model": args.model, "tokenizer": args.tokenizer or args.model,
        "chat_template": "tokenizer default (vllm LLM.chat, add_generation_prompt=True)",
        "tasks": args.tasks, "few_shot_tasks": sorted(set(args.tasks) & FEW_SHOT_TASKS),
        "temperature": args.temperature, "top_p": args.top_p, "max_new_tokens": args.max_new_tokens,
        "max_model_len": args.max_model_len, "fallback_max_model_len": args.fallback_max_model_len,
        "seed": args.seed, "dtype": "auto", "tensor_parallel_size": 1, "gpu_memory_utilization": 0.90,
        "scorer": args.scorer, "scorer_revision": args.scorer_revision, "eval_type": "node",
        "limit": args.limit,
    }
    settings_path = args.output / "settings.json"
    if args.score_only:
        # Rescore a finished run: its generation settings come from its settings.json.
        if not settings_path.exists():
            p.error(f"--score-only needs {settings_path}")
        settings = load_json(settings_path)
        if (settings["scorer"], settings["scorer_revision"]) != (args.scorer, args.scorer_revision):
            p.error("--score-only must use the run's scorer and scorer revision")
        args.tasks, args.limit = settings["tasks"], settings["limit"]
    else:
        args.output.mkdir(parents=True, exist_ok=True)
        if settings_path.exists() and load_json(settings_path) != settings:
            raise SystemExit(f"{settings_path} records different settings; use a new --output")
        save_json(settings_path, settings)
    print(json.dumps(settings, indent=2), flush=True)

    if args.merge_shards:
        merge_shards(args)
    elif not args.score_only:
        name = (lambda t: f"{t}.shard{args.shard_id}.json") if args.num_shards > 1 else (lambda t: f"{t}.json")
        generate(args, lambda t: args.output / "pred" / name(t))
    if args.generate_only:
        return
    score(args, settings)


if __name__ == "__main__":
    main()
