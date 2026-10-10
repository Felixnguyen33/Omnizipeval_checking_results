#!/usr/bin/env bash
# OmniZipEval LiveCodeBench, DeepSeek-R1-Distill family (reasoning, <think>).
# Dense parents: DeepSeek-R1-Distill-Qwen-1.5B, -Qwen-7B, -Llama-8B, and every
# compressed checkpoint made from them (GBLM-Pruner / Wanda / SparseGPT 50%,
# AWQ, GPTQ-Int4, SmoothQuant W8A8).
#
#   bash scripts/omnizipeval/deepseek_r1_distill.sh                   # dense Qwen-7B
#   MODEL=deepseek-ai/DeepSeek-R1-Distill-Llama-8B GPUS=0,1 bash scripts/omnizipeval/deepseek_r1_distill.sh
#   MODEL=/path/to/ds_r1_qwen_1_5b_sparsegpt50 PARENT=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
#     GPUS=0 bash scripts/omnizipeval/deepseek_r1_distill.sh
#
# The tokenizer defaults to the dense parent's (TOKENIZER=PARENT), so every
# checkpoint of a cohort is decoded with the same tokenizer.
#
# Inputs: MODEL PARENT TOKENIZER GPUS TP_SIZE DTYPE OUTPUT PYTHON (see _run.sh).
set -euo pipefail

readonly PROTOCOL="lcb_v4_deepseek_r1_distill"
readonly DEFAULT_MODEL="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
readonly PARENTS=("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B" "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B" "deepseek-ai/DeepSeek-R1-Distill-Llama-8B")
readonly PROMPT_STYLE="DeepSeekR1"          # LiveCodeBench DeepSeek-R1 prompt (lm_styles.py)

readonly SCENARIO="codegeneration"
readonly RELEASE_VERSION="release_v4"       # contests 2023-05-07 .. 2024-10-05
readonly DATASET_REVISION="0fe84c3912ea0c4d4a78037083943e8f0c4dd505"
readonly EXPECTED_PROBLEMS=713              # no --start_date / --end_date filter

readonly N=1
readonly TEMPERATURE=0.2
readonly TOP_P=0.95
readonly MAX_TOKENS=32768
readonly STOP="<｜end▁of▁sentence｜>"       # DeepSeek EOS; "###" would cut the reasoning
readonly MAX_MODEL_LEN=40960
readonly TIMEOUT=6                          # seconds per test case

source "$(dirname "${BASH_SOURCE[0]}")/_run.sh"
