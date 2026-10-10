#!/usr/bin/env bash
# OmniZipEval LiveCodeBench, Qwen2.5-Instruct family (non-reasoning).
# Dense parents: Qwen2.5-3B/7B/32B-Instruct, and every compressed checkpoint
# made from them (GBLM-Pruner / Wanda / SparseGPT 50%, AWQ, GPTQ-Int4,
# SmoothQuant W8A8).
#
#   bash scripts/omnizipeval/qwen25_instruct.sh                       # dense 7B
#   MODEL=Qwen/Qwen2.5-3B-Instruct GPUS=0,1 bash scripts/omnizipeval/qwen25_instruct.sh
#   MODEL=/path/to/qwen25_7b_wanda50 PARENT=Qwen/Qwen2.5-7B-Instruct \
#     GPUS=0 bash scripts/omnizipeval/qwen25_instruct.sh
#   MODEL=Qwen/Qwen2.5-32B-Instruct-AWQ PARENT=Qwen/Qwen2.5-32B-Instruct \
#     DTYPE=float16 GPUS=0,1,2,3 bash scripts/omnizipeval/qwen25_instruct.sh
#
# Inputs: MODEL PARENT TOKENIZER GPUS TP_SIZE DTYPE OUTPUT PYTHON (see _run.sh).
set -euo pipefail

readonly PROTOCOL="lcb_v4_qwen25_instruct"
readonly DEFAULT_MODEL="Qwen/Qwen2.5-7B-Instruct"
readonly PARENTS=("Qwen/Qwen2.5-3B-Instruct" "Qwen/Qwen2.5-7B-Instruct" "Qwen/Qwen2.5-32B-Instruct")
readonly PROMPT_STYLE="CodeQwenInstruct"   # LiveCodeBench ChatML prompt (lm_styles.py)

readonly SCENARIO="codegeneration"
readonly RELEASE_VERSION="release_v4"       # contests 2023-05-07 .. 2024-10-05
readonly DATASET_REVISION="0fe84c3912ea0c4d4a78037083943e8f0c4dd505"
readonly EXPECTED_PROBLEMS=713              # no --start_date / --end_date filter

readonly N=10
readonly TEMPERATURE=0.2
readonly TOP_P=0.95
readonly MAX_TOKENS=2000
readonly STOP="###"                         # LiveCodeBench parser default
readonly MAX_MODEL_LEN=16384
readonly TIMEOUT=6                          # seconds per test case

source "$(dirname "${BASH_SOURCE[0]}")/_run.sh"
