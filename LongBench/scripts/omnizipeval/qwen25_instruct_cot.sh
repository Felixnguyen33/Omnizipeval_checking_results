#!/usr/bin/env bash
# LongBench v2, two-stage CoT, Qwen2.5-Instruct family (non-thinking).
# Dense models: Qwen2.5-3B-Instruct, Qwen2.5-7B-Instruct, and their compressed variants
# (GBLM-Pruner, Wanda, SparseGPT, AWQ, GPTQ-Int4, SmoothQuant).
#
# Usage (from anywhere):
#   bash scripts/omnizipeval/qwen25_instruct_cot.sh                                  # dense Qwen2.5-3B-Instruct
#   MODEL=Qwen/Qwen2.5-7B-Instruct GPUS=1 bash scripts/omnizipeval/qwen25_instruct_cot.sh
#   MODEL=/ckpt/qwen25_7b_wanda50 PARENT=Qwen/Qwen2.5-7B-Instruct NAME=qwen25_7b_wanda50 \
#     bash scripts/omnizipeval/qwen25_instruct_cot.sh
#
# Env: MODEL (HF id or local path; default dense Qwen/Qwen2.5-3B-Instruct), PARENT (tokenizer/chat
# template; default MODEL, use the dense model for compressed checkpoints), NAME (run name),
# GPUS (default 0), TP/PP (default #GPUS / 1), OUTPUT (default outputs/qwen25_instruct_cot/NAME),
# PYTHON, PORT, DTYPE (default bfloat16; only for kernels that cannot run BF16).
set -euo pipefail
SCRIPT_TAG="qwen25_instruct_cot"
DEFAULT_MODEL="Qwen/Qwen2.5-3B-Instruct"
DEFAULT_GPUS="0"
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
common_env

# ---- protocol (hard-coded) --------------------------------------------------
MODE="cot"                      # pred.py --cot: prompts/0shot_cot.txt then prompts/0shot_cot_ans.txt
MAX_INPUT_TOKENS=30000          # middle truncation: first 15000 + last 15000 prompt tokens (PARENT tokenizer)
MAX_MODEL_LEN=32768             # native Qwen2.5 context, no RoPE scaling
COT_MAX_NEW_TOKENS=1024         # stage 1 (upstream pred.py value)
ANS_MAX_NEW_TOKENS=128          # stage 2 (upstream pred.py value)
TEMPERATURE=0.1                 # upstream pred.py value
TOP_P=1.0
TOP_K=20                        # = Qwen2.5-Instruct generation_config.json value, sent explicitly
REPETITION_PENALTY=1.05         # = Qwen2.5-Instruct generation_config.json value, sent explicitly
THINKING="none (Qwen2.5-Instruct has no thinking mode; default chat template, user turn only)"
N_PROC=8                        # concurrent client processes
DTYPE="${DTYPE:-bfloat16}"
GPU_MEM=0.92
# -----------------------------------------------------------------------------

tee "$OUTPUT/settings.txt" <<EOF
script            : $SCRIPT_TAG
model             : $MODEL
parent/tokenizer  : $PARENT
name              : $NAME
dataset           : $DATASET_ID @ $DATASET_REVISION, split train, $DATASET_SIZE questions
mode              : two-stage CoT (prompts/0shot_cot.txt -> prompts/0shot_cot_ans.txt, \\boxed{} answer)
input truncation  : $MAX_INPUT_TOKENS tokens, keep first/last $((MAX_INPUT_TOKENS / 2))
max_model_len     : $MAX_MODEL_LEN
max_new_tokens    : cot $COT_MAX_NEW_TOKENS / answer $ANS_MAX_NEW_TOKENS
sampling          : temperature $TEMPERATURE, top_p $TOP_P, top_k $TOP_K, repetition_penalty $REPETITION_PENALTY, no seed
thinking          : $THINKING
backend           : vLLM OpenAI server, dtype $DTYPE, --generation-config vllm, TP $TP x PP $PP, GPUs $GPUS, gpu_mem $GPU_MEM, $N_PROC client procs
output            : $OUTPUT
EOF

start_server --dtype "$DTYPE" --max-model-len "$MAX_MODEL_LEN" --gpu-memory-utilization "$GPU_MEM" \
  --generation-config vllm --trust-remote-code

run_pred "${NAME}_cot.jsonl" \
  --model "$NAME" --model_path "$MODEL" --tokenizer "$PARENT" --cot \
  --max_len "$MAX_INPUT_TOKENS" \
  --cot_max_new_tokens "$COT_MAX_NEW_TOKENS" --ans_max_new_tokens "$ANS_MAX_NEW_TOKENS" \
  --temperature "$TEMPERATURE" --top_p "$TOP_P" --top_k "$TOP_K" --repetition_penalty "$REPETITION_PENALTY" \
  --dataset_revision "$DATASET_REVISION" --n_proc "$N_PROC"
