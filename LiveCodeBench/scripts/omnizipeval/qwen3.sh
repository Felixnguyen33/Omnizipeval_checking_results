#!/usr/bin/env bash
# OmniZipEval LiveCodeBench, Qwen3 family (thinking mode, the model's default).
# Dense parents: Qwen3-8B, Qwen3-30B-A3B, and every compressed checkpoint made
# from them (GBLM-Pruner / Wanda / SparseGPT 50%, AWQ, GPTQ-Int4,
# SmoothQuant W8A8).
#
#   bash scripts/omnizipeval/qwen3.sh                                 # dense Qwen3-8B
#   MODEL=Qwen/Qwen3-30B-A3B GPUS=0,1,2,3 GPU_MEMORY_UTILIZATION=0.75 bash scripts/omnizipeval/qwen3.sh
#   MODEL=Qwen/Qwen3-8B-AWQ PARENT=Qwen/Qwen3-8B GPUS=0,1 bash scripts/omnizipeval/qwen3.sh
#
# Inputs: MODEL PARENT TOKENIZER GPUS TP_SIZE DTYPE OUTPUT PYTHON (see _run.sh).
set -euo pipefail

readonly PROTOCOL="lcb_v4_qwen3"
readonly DEFAULT_MODEL="Qwen/Qwen3-8B"
readonly PARENTS=("Qwen/Qwen3-8B" "Qwen/Qwen3-30B-A3B")
readonly PROMPT_STYLE="CodeQwenInstruct"   # LiveCodeBench ChatML prompt (lm_styles.py)

readonly SCENARIO="codegeneration"
readonly RELEASE_VERSION="release_v4"       # contests 2023-05-07 .. 2024-10-05
readonly DATASET_REVISION="0fe84c3912ea0c4d4a78037083943e8f0c4dd505"
readonly EXPECTED_PROBLEMS=713              # no --start_date / --end_date filter

readonly N=1
readonly TEMPERATURE=0.6
readonly TOP_P=0.95
readonly MAX_TOKENS=32768
readonly STOP="<|im_end|>"                  # Qwen EOS; "###" would cut the reasoning
readonly MAX_MODEL_LEN=40960
readonly TIMEOUT=6                          # seconds per test case

source "$(dirname "${BASH_SOURCE[0]}")/_run.sh"
