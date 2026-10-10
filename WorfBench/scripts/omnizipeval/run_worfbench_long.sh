#!/usr/bin/env bash
# WorfBench, OmniZipEval protocol with the DeepSeek-R1 generation length (32,768 new tokens,
# 40,960-token context, no shorter-context retry). Same tasks, prompts, decoding and scoring as
# run_worfbench.sh; only the length limits differ.
# Intended for reasoning models (DeepSeek-R1-Distill) that need a long generation budget.
#
# Usage (env vars): MODEL (default deepseek-ai/DeepSeek-R1-Distill-Qwen-7B), TOKENIZER, GPUS
# (default 0; one data-parallel shard per GPU), OUTPUT, PYTHON - see run_worfbench.sh.
#
#   GPUS=0,1,2,3,4,5,6 bash scripts/omnizipeval/run_worfbench_long.sh
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

CONFIG="run_worfbench_long.sh (32,768 new tokens)"
MODEL="${MODEL:-deepseek-ai/DeepSeek-R1-Distill-Qwen-7B}"
TOKENIZER="${TOKENIZER:-}"
GPUS="${GPUS:-0}"
OUTPUT="${OUTPUT:-$PWD/worfbench_outputs_32k/${MODEL//\//__}}"

MAX_NEW_TOKENS=32768
MAX_MODEL_LEN=40960      # longest prompt is 5,161 tokens, so 32,768 new tokens always fit
FALLBACK_MAX_MODEL_LEN=0 # a long-context run must not silently fall back to 8,192

echo_settings
run_config
