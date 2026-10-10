#!/usr/bin/env bash
# LongBench v2, two-stage CoT, DeepSeek-R1-Distill family (always thinking).
# Dense models: DS-R1-Distill-Llama-8B, -Qwen-1.5B, -Qwen-7B, and their compressed variants.
#
# Usage:
#   bash scripts/omnizipeval/deepseek_r1_distill_cot.sh                               # dense DeepSeek-R1-Distill-Llama-8B
#   MODEL=deepseek-ai/DeepSeek-R1-Distill-Qwen-7B bash scripts/omnizipeval/deepseek_r1_distill_cot.sh
#   MODEL=/ckpt/ds_llama8b_awq PARENT=deepseek-ai/DeepSeek-R1-Distill-Llama-8B NAME=ds_llama8b_awq \
#     bash scripts/omnizipeval/deepseek_r1_distill_cot.sh
#
# Env: MODEL (default dense deepseek-ai/DeepSeek-R1-Distill-Llama-8B), PARENT (tokenizer/chat template;
# default MODEL; use the dense model for compressed checkpoints so every variant gets the same
# template, including the "<think>" generation prefix), NAME, GPUS (default 0), TP/PP, OUTPUT,
# PYTHON, PORT, DTYPE (default bfloat16).
set -euo pipefail
SCRIPT_TAG="deepseek_r1_distill_cot"
DEFAULT_MODEL="deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
DEFAULT_GPUS="0"
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
common_env

# ---- protocol (hard-coded) --------------------------------------------------
MAX_INPUT_TOKENS=96000          # middle truncation: first 48000 + last 48000 prompt tokens (PARENT tokenizer)
MAX_MODEL_LEN=131072            # native R1-Distill context (96K input + 32K output fits)
MAX_NEW_TOKENS=2048             # pred.py -t 2048: BOTH stages; output file gets the _t2048 suffix
TEMPERATURE=0.6                 # DeepSeek-R1 model card
TOP_P=0.95                      # DeepSeek-R1 model card
THINKING="always on (the R1-Distill chat template opens <think>); text before the first </think> is ignored by extract_answer"
N_PROC=8
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
max_new_tokens    : $MAX_NEW_TOKENS per stage (cot and answer)
sampling          : temperature $TEMPERATURE, top_p $TOP_P, top_k off, no repetition penalty, no seed
thinking          : $THINKING
backend           : vLLM OpenAI server, dtype $DTYPE, --generation-config vllm, TP $TP x PP $PP, GPUs $GPUS, gpu_mem $GPU_MEM, $N_PROC client procs
output            : $OUTPUT
EOF

start_server --dtype "$DTYPE" --max-model-len "$MAX_MODEL_LEN" --gpu-memory-utilization "$GPU_MEM" \
  --generation-config vllm --trust-remote-code

run_pred "${NAME}_cot_t${MAX_NEW_TOKENS}.jsonl" \
  --model "$NAME" --model_path "$MODEL" --tokenizer "$PARENT" --cot \
  --max_len "$MAX_INPUT_TOKENS" --max_new_tokens "$MAX_NEW_TOKENS" \
  --temperature "$TEMPERATURE" --top_p "$TOP_P" \
  --dataset_revision "$DATASET_REVISION" --n_proc "$N_PROC"
