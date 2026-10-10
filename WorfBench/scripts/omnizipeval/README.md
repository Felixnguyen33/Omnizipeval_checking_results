# WorfBench for OmniZipEval

This folder holds the OmniZipEval scripts for WorfBench node-level precision / recall / F1.
The rest of this directory is upstream WorfBench.

## Source

- Upstream: [zjunlp/WorfBench](https://github.com/zjunlp/WorfBench), commit
  `38a6455557128dffceae9a08ffd745f04a4a9d03` ("Update README.md").
- Copied with `rsync`, without `.git` and `__pycache__` (upstream tracks three cpython-310 `.pyc`
  files; they are left out).
- The data is the bundled `gold_traj/<task>/graph_eval.json` of that commit. Nothing is downloaded.

| Task | Examples | Shots | sha256 of `gold_traj/<task>/graph_eval.json` |
|---|---:|---|---|
| alfworld | 312 | 2 | `5f983a9da7cf3389015f418ca934a30046f296d7b7958e6beacc43539b32a239` |
| intercodesql | 500 | 0 | `f8ed99c5653c73440b9416bf8c0525b4897be3d17dcfb106b44db65fb5fa379c` |
| lumos | 489 | 2 | `89b422175c2fca97a99b18ef0f57172647c40377c5eedf7acd82cf73a31b1436` |
| os | 20 | 2 | `7d64a0e3060493524d527935b8f9eb9d32b90901c6ad2dc784d61add740d2219` |
| seal_tools | 223 | 0 | `bbc3d9b50fe7f76876726b2295dc8ba0b0e9325c33ba277b81d64e4f6d460123` |
| toolalpaca | 93 | 2 | `4d48c72fa9ee7ef1420a7bf2d859f81fa42d146c12a1a9f23250c6a6f5436bcd` |
| toolbench | 114 | 2 | `bc4890b27d523aa0ffbc6a88ce11e26265010b20ca0b2fc010762c385f516381` |
| webshop | 133 | 2 | `eada0bc23f960cfab968f5ba549d9b700a1a86dded1a84136a1a0e15a2140dde` |
| wikihow (not used) | 262 | 2 | `681a6d2fa53df166558b520f5f704e88d54978478d058bf9521f6e787c8b4f60` |

`intercodesql` and `seal_tools` have no entry in `prompts/eval_prompt.py:two_shot_example`, so
they are zero-shot.

## Changes from upstream

| File | Change | Why |
|---|---|---|
| `node_eval.py` | The node-line regex `\d+[:.] (.+)` was compiled inline twice (`workflow_to_node_list`, `workflow_to_graph_list`). It is now one module-level `NODE_PATTERN` with the same default. | OmniZipEval parses node lines with `\d+[:.]\s*(.+)`, which also accepts `1:text` / `1.text` with no space. `generate.py` sets `NODE_PATTERN`. Upstream behaviour is unchanged when the module is used on its own. |
| `scripts/omnizipeval/` | New: `generate.py`, `common.sh`, `run_worfbench.sh`, `run_worfbench_long.sh`, `rescore_strict_parser.sh`, `requirements.txt`, this README. | Upstream has no offline generator: `node_eval.py --task gen_workflow` posts one request at a time to an OpenAI-style server (`LLM/localLLM.py`, defaults temperature 0.5, top_p 1, 1,024 tokens). |

Every other file is byte-identical to upstream.

`generate.py` does the following:

1. Builds the prompt with upstream `node_eval.build_message`.
2. Generates with `vllm.LLM.chat`.
3. Scores with upstream `node_eval.eval_workflow(..., "node", ...)`.

It also patches two things at run time and leaves upstream files as they are:

- **Scorer.** `SentenceTransformer` loads the pinned revision on the chosen device and caches each
  string's embedding. Upstream re-encodes the same node for each of up to 20 topological sorts.
  The scores do not change.
- **Empty workflows.** `t_eval_nodes` returns 0 when the prediction or the gold workflow has no
  parsed nodes. Upstream would otherwise hand the encoder an empty list. Unparsed workflows
  already count as 0 in upstream `eval_workflow`.

## Install and run

| Script | Use | Tokens / context |
|---|---|---|
| `run_worfbench.sh` | Default setting for every model and compressed variant | 1,024 / 16,384 (8,192 retry) |
| `run_worfbench_long.sh` | Long-generation variant for reasoning models (DeepSeek-R1-Distill) | 32,768 / 40,960 (no retry) |
| `rescore_strict_parser.sh` | Optional: rescores a finished run with upstream's strict parser (parser-sensitivity check) | none (scoring only) |

Environment variables:

- `MODEL`: HF id or local path, dense or compressed. The default is the dense original:
  `Qwen/Qwen2.5-7B-Instruct`, or `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B` for the long script.
- `TOKENIZER`: optional. The default is the model's own tokenizer and chat template. Set it to the
  dense parent for compressed checkpoints whose repo lacks a correct chat template.
- `GPUS`: comma-separated GPU ids. With more than one GPU, the script runs one data-parallel
  shard per GPU, then merges and scores.
- `OUTPUT`: run directory.
- `PYTHON`: interpreter (see `requirements.txt`).

```bash
pip install -r scripts/omnizipeval/requirements.txt
MODEL=Qwen/Qwen2.5-3B-Instruct GPUS=0 OUTPUT=runs/worfbench/qwen25_3b/base \
  bash scripts/omnizipeval/run_worfbench.sh
MODEL=Qwen/Qwen2.5-3B-Instruct-AWQ GPUS=1 OUTPUT=runs/worfbench/qwen25_3b/awq \
  bash scripts/omnizipeval/run_worfbench.sh
GPUS=0,1,2,3,4,5,6 OUTPUT=runs/worfbench_32k/ds_r1_qwen_7b/base \
  bash scripts/omnizipeval/run_worfbench_long.sh
OUTPUT=runs/worfbench/qwen25_3b/base bash scripts/omnizipeval/rescore_strict_parser.sh
```

## Metric

The score is `OUTPUT/summary_optional_space.json` → `average_x100`. Precision, recall and F1 are
computed per example. For each example, the topological sort of the gold graph with the highest F1
is used. The per-example values are averaged within each task, and the eight tasks are averaged
with equal weight. Tables show this ×100 with one decimal.

## Files written in `OUTPUT`

- `settings.json`: every setting of the run (resuming in the same directory requires the same settings).
- `pred/<task>.json`: predictions in upstream format (`query`, `workflow`) plus `finish_reason` and
  `n_tokens`; `pred/<task>.shard<i>.json` for multi-GPU runs, merged before scoring.
- `runtime_shard<i>.json`: the `max_model_len` vLLM loaded with.
- `eval_<parser>/<task>_node.json`: upstream score file per task.
- `summary_<parser>.json`: per-task scores and the equal-weight 8-task mean.
- `logs/run.log` (one GPU) or `logs/shard<i>.log` + `logs/score.log` (several GPUs): console output.

## Settings

| Setting | Value |
|---|---|
| Tasks | alfworld, intercodesql, lumos, os, seal_tools, toolalpaca, toolbench, webshop (full gold files, 1,884 examples) |
| Shots | 2-shot (upstream `two_shot_example`), 0-shot for intercodesql and seal_tools |
| Prompt | upstream `build_message`: system turn, then the two-shot turns, then the user turn prefixed "Now it's your turn." |
| Chat template | model's own tokenizer (or `TOKENIZER`), `vllm.LLM.chat`, `add_generation_prompt=True` |
| Temperature / top_p | 0.0 / 0.95. Explicit `SamplingParams`, so the model's `generation_config.json` is not applied and top_k and repetition_penalty stay at vLLM defaults. |
| Max new tokens | 1,024. The long script uses 32,768. |
| Max model len | 16,384, with a retry at 8,192 and `enforce_eager` if the load fails. The long script uses 40,960 with no retry. |
| Seed | 0 (vLLM default; irrelevant at temperature 0) |
| dtype / TP / memory | auto / 1 / 0.90, `trust_remote_code=True` |
| Node parser | `optional_space`: `\d+[:.]\s*(.+)` (`strict_space` = upstream `\d+[:.] (.+)`, only via `rescore_strict_parser.sh`) |
| Scorer | `sentence-transformers/all-mpnet-base-v2` @ `e8c3b32edf5434bc2275fc9bab85f82640a19130`, node eval, match threshold 0.6, first 20 topological sorts (`evaluator/graph_evaluator.py`) |
| Software | vllm 0.27.1 (V1 engine), torch 2.13.0, transformers 5.15.1, sentence-transformers 6.0.1, networkx 3.6.1 |

## Models

- Dense models: Qwen2.5-3B-Instruct, Qwen2.5-7B-Instruct, DeepSeek-R1-Distill-Llama-8B,
  DeepSeek-R1-Distill-Qwen-1.5B, DeepSeek-R1-Distill-Qwen-7B.
- Variants of each: GBLM-Pruner, Wanda, SparseGPT (50% unstructured), AWQ, GPTQ-Int4,
  SmoothQuant W8A8.
- Pass any compressed checkpoint (Hub id or local directory) as `MODEL`.
