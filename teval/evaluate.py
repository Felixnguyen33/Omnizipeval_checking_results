#!/usr/bin/env python3
"""T-Eval (English v2, all eight subsets) for a dense model or a compressed variant of it.

Every protocol setting is a constant below; the command line only names the checkpoints, the
output folder and the GPU count. The tokenizer and chat template always come from --parent.

    python teval/download_data.py                       # once
    CUDA_VISIBLE_DEVICES=0 python teval/evaluate.py \
        --model Qwen/Qwen2.5-3B-Instruct --revision <sha> \
        --parent Qwen/Qwen2.5-3B-Instruct --parent-revision <sha> \
        --output /path/outside/repo/teval/qwen2.5-3b-instruct
"""
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "common"))
import fairness  # noqa: E402
import scoring  # noqa: E402
from download_data import DATA_REPO, DATA_REVISION, verify as verify_data  # noqa: E402

# ----------------------------------------------------------------------------- protocol constants
PROTOCOL_ID = "teval-en-v2-checked-v1"
BACKEND = "vllm"
DTYPE = "bfloat16"
CONTEXT_TOKENS = 16384        # longest English v2 prompt is ~8.2k tokens (Qwen2.5 tokenizer); no prompt is truncated
MAX_NEW_TOKENS = 4096         # leaves room for DeepSeek-R1 reasoning; prompt + budget always fits CONTEXT_TOKENS
SEED = 0
SAMPLING = {"n": 1, "temperature": 0.0, "top_p": 1.0, "top_k": -1, "min_p": 0.0, "repetition_penalty": 1.0,
            "presence_penalty": 0.0, "frequency_penalty": 0.0, "max_tokens": MAX_NEW_TOKENS, "seed": SEED,
            "skip_special_tokens": True}
ENGINE = {"dtype": DTYPE, "max_model_len": CONTEXT_TOKENS, "gpu_memory_utilization": 0.85, "max_num_seqs": 256,
          "max_num_batched_tokens": 16384, "enable_prefix_caching": False, "seed": SEED,
          "generation_config": "vllm",  # ignore every checkpoint's generation_config.json defaults
          "trust_remote_code": False}
BATCHING = "all prompts in one vLLM generate call in fixed item order; continuous batching (max_num_seqs=256)"
VERSIONED_PACKAGES = ("vllm", "torch", "transformers", "tokenizers", "sentence-transformers", "mmengine",
                      "networkx", "numpy", "huggingface-hub", "safetensors")
SOURCES = [ROOT / "evaluate.py", ROOT / "scoring.py", ROOT / "download_data.py", ROOT.parent / "common" / "fairness.py",
           *sorted((ROOT / "third_party" / "teval" / "evaluators").glob("*_evaluator.py")),
           ROOT / "third_party" / "teval" / "utils" / "format_load.py",
           ROOT / "third_party" / "teval" / "utils" / "template.py",
           ROOT / "third_party" / "teval" / "utils" / "convert_results.py",
           ROOT / "third_party" / "teval" / "schema.py"]


def parent_stop_token_ids(tokenizer, parent, parent_revision):
    """EOS ids of the parent: its tokenizer EOS plus generation_config.json eos_token_id (same for the cohort)."""
    from transformers import GenerationConfig
    ids = {tokenizer.eos_token_id}
    try:
        eos = GenerationConfig.from_pretrained(parent, revision=parent_revision).eos_token_id
        ids |= set(eos if isinstance(eos, list) else [eos])
    except OSError:
        pass
    return sorted(i for i in ids if i is not None)


def build_prompts(tokenizer, items):
    for item in items:
        text = tokenizer.apply_chat_template(scoring.messages(item["datum"]), add_generation_prompt=True,
                                             tokenize=False)
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        if len(ids) + MAX_NEW_TOKENS > CONTEXT_TOKENS:
            raise SystemExit(f"{item['id']}: {len(ids)} prompt tokens + {MAX_NEW_TOKENS} > {CONTEXT_TOKENS}")
        item.update(prompt_ids=ids, prompt_sha256=scoring.token_ids_sha256(ids),
                    opens_think=scoring.template_opens_think(text))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    fairness.add_model_arguments(p)
    p.add_argument("--tensor-parallel-size", type=int, default=1,
                   help="GPUs per model (hardware only; recorded in manifest.json and run_info.json)")
    p.add_argument("--smoke", type=int, metavar="N",
                   help="Pipeline test on the first N items of each subset. Recorded in the protocol, so a smoke "
                        "run never pairs with a full run. For pipeline tests only.")
    args = p.parse_args()
    if args.smoke is not None and args.smoke < 1:
        p.error("--smoke must be positive")
    fairness.identity(args.model, args.revision)
    parent_id = fairness.identity(args.parent, args.parent_revision)
    data_sha256 = verify_data()

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.parent, revision=args.parent_revision, trust_remote_code=False)
    stop_ids = parent_stop_token_ids(tokenizer, args.parent, args.parent_revision)
    items = scoring.load_items(args.smoke)
    build_prompts(tokenizer, items)

    protocol = {
        "id": PROTOCOL_ID,
        "dataset": {"repo": DATA_REPO, "revision": DATA_REVISION, "files_sha256": data_sha256,
                    "split": f"first {args.smoke} items per subset (smoke)" if args.smoke else "full",
                    "language": "en", "version": "v2"},
        "subsets": list(scoring.SUBSETS),
        "prompt": {"source": "T-Eval origin_prompt messages, parent tokenizer apply_chat_template("
                             "add_generation_prompt=True), encoded with add_special_tokens=False",
                   "role_map": {"function": "user"}, "meta_template": "parent chat template (no lagent template)",
                   "parent_tokenizer": parent_id,
                   "chat_template_sha256": hashlib.sha256((tokenizer.chat_template or "").encode()).hexdigest(),
                   "vocab_sha256": hashlib.sha256(json.dumps(tokenizer.get_vocab(), sort_keys=True).encode()).hexdigest(),
                   "prompt_set_sha256": hashlib.sha256("".join(i["prompt_sha256"] for i in items).encode()).hexdigest()},
        "decoding": {**SAMPLING, "greedy": True, "stop_token_ids": stop_ids},
        "max_new_tokens": MAX_NEW_TOKENS, "context_tokens": CONTEXT_TOKENS, "truncation": "none (refused)",
        "backend": BACKEND, "engine": ENGINE, "dtype": DTYPE, "batching": BATCHING, "seeds": SEED,
        "postprocess": scoring.POSTPROCESS_RULE,
        "scorer": {"code": scoring.UPSTREAM_TEVAL, "embedding_model": scoring.EMBEDDING_MODEL,
                   "embedding_revision": scoring.EMBEDDING_REVISION, "embedding_device": scoring.EMBEDDING_DEVICE,
                   "overall": scoring.OVERALL_RULE},
        "api_judges": None,
        "source_sha256": fairness.source_hashes(*SOURCES),
        "versions": fairness.package_versions(*VERSIONED_PACKAGES),
    }
    if args.smoke:
        protocol["smoke"] = args.smoke
    scoring.embedding_model_path()  # resolve the pinned scorer before spending GPU time

    manifest = fairness.manifest_from_args(args, "teval", protocol)  # checkpoint checks; refuses existing output
    out = args.output
    hardware = {"tensor_parallel_size": args.tensor_parallel_size,
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")}

    from vllm import LLM, SamplingParams
    llm = LLM(model=args.model, revision=args.revision, tokenizer=args.parent, tokenizer_revision=args.parent_revision,
              tensor_parallel_size=args.tensor_parallel_size, **ENGINE)
    import torch
    hardware["gpus"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    hardware["vllm_quantization"] = llm.llm_engine.model_config.quantization
    manifest.data["hardware"] = hardware
    manifest.write()

    started = time.time()
    outputs = llm.generate([{"prompt_token_ids": i["prompt_ids"]} for i in items],
                           SamplingParams(**SAMPLING, stop_token_ids=stop_ids), use_tqdm=True)
    generation_seconds = time.time() - started

    predictions = {s["key"]: {} for s in scoring.SUBSETS}
    stats = {"finish_length": 0, "reasoning_unfinished": 0, "output_tokens": 0}
    with (out / "generations.jsonl").open("w") as stream:
        for item, output in zip(items, outputs):
            completion = output.outputs[0]
            prediction, closed = scoring.extract_prediction(completion.text, item["opens_think"])
            row = {"id": item["id"], "subset": item["subset"], "key": item["key"],
                   "prompt_sha256": item["prompt_sha256"], "prompt_tokens": len(item["prompt_ids"]),
                   "opens_think": item["opens_think"], "raw": completion.text,
                   "output_tokens": len(completion.token_ids), "finish_reason": completion.finish_reason,
                   "reasoning_closed": closed, "prediction": prediction}
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            predictions[item["subset"]][item["key"]] = {**item["datum"], "prediction": prediction}
            stats["finish_length"] += completion.finish_reason == "length"
            stats["reasoning_unfinished"] += closed is False
            stats["output_tokens"] += len(completion.token_ids)
    (out / "predictions").mkdir()
    for key, rows in predictions.items():  # upstream test.py output format, read by the upstream evaluators
        (out / "predictions" / f"{key}.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1) + "\n")

    results = scoring.score_folder(out / "predictions")
    (out / "scores_by_subset.json").write_text(json.dumps(results, indent=2) + "\n")  # upstream <name>_-1.json layout
    scores = scoring.capabilities(results)
    (out / "run_info.json").write_text(json.dumps({"hardware": hardware, "generation_seconds": generation_seconds,
                                                   "generation_stats": stats, "n_items": len(items)}, indent=2) + "\n")
    manifest.data["generation_stats"] = stats
    manifest.complete(scores, [i["id"] for i in items])
    print(json.dumps({"scores": {k: round(v, 1) for k, v in scores.items()}, "generation_stats": stats}, indent=2))


if __name__ == "__main__":
    main()
