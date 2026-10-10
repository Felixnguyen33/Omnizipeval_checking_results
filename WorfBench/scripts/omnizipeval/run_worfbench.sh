#!/usr/bin/env bash
# WorfBench, OmniZipEval protocol (node-level precision / recall / F1).
# Eight tasks, node-level P/R/F1, greedy decoding, 1,024 new tokens, 16,384-token context.
# Default setting for every model; run_worfbench_long.sh is the long-generation variant.
#
# Usage (env vars):
#   MODEL      HF id or local path, dense or compressed   (default Qwen/Qwen2.5-7B-Instruct)
#   TOKENIZER  optional tokenizer / chat template source   (default: the model's own. Set it to the
#              dense parent for compressed checkpoints whose repo lacks a correct chat template.)
#   GPUS       comma-separated GPU ids; >1 GPU = one data-parallel shard per GPU (default 0)
#   OUTPUT     run directory (default ./worfbench_outputs/<MODEL with / -> __>)
#   PYTHON     interpreter with vllm, sentence-transformers, networkx, tqdm, requests (default python)
#
#   MODEL=Qwen/Qwen2.5-3B-Instruct-AWQ GPUS=0 bash scripts/omnizipeval/run_worfbench.sh
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

CONFIG="run_worfbench.sh (1,024 new tokens)"
MODEL="${MODEL:-Qwen/Qwen2.5-7B-Instruct}"
TOKENIZER="${TOKENIZER:-}"
GPUS="${GPUS:-0}"
OUTPUT="${OUTPUT:-$PWD/worfbench_outputs/${MODEL//\//__}}"

MAX_NEW_TOKENS=1024
MAX_MODEL_LEN=16384
FALLBACK_MAX_MODEL_LEN=8192

echo_settings
run_config
