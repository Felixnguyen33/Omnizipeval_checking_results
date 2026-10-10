#!/usr/bin/env bash
# LongBench v2, two-stage CoT, Qwen3 MoE (thinking on).
# Dense model: Qwen3-30B-A3B, and its compressed variants (GBLM-Pruner, Wanda, SparseGPT,
# AWQ, GPTQ-Int4, SmoothQuant).
#
# Usage:
#   bash scripts/omnizipeval/qwen3_moe_cot.sh                                         # dense Qwen3-30B-A3B on GPUs 0-3
#   MODEL=/ckpt/qwen3_30b_a3b_gptq PARENT=Qwen/Qwen3-30B-A3B NAME=qwen3_30b_a3b_gptq GPUS=0,1,2 TP=1 PP=3 \
#     bash scripts/omnizipeval/qwen3_moe_cot.sh
#
# Env: MODEL (default dense Qwen/Qwen3-30B-A3B), PARENT (tokenizer/chat template; default MODEL; use the
# dense model for compressed checkpoints), NAME, GPUS (default 0,1,2,3), TP/PP (default #GPUS / 1),
# OUTPUT, PYTHON, PORT, DTYPE (default bfloat16).
set -euo pipefail
SCRIPT_TAG="qwen3_moe_cot"
DEFAULT_MODEL="Qwen/Qwen3-30B-A3B"
DEFAULT_GPUS="0,1,2,3"
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
common_env

# ---- protocol (hard-coded) --------------------------------------------------
MAX_INPUT_TOKENS=30000          # middle truncation: first 15000 + last 15000 prompt tokens (PARENT tokenizer)
MAX_MODEL_LEN=40960             # Qwen3-30B-A3B configured context, no RoPE scaling
COT_MAX_NEW_TOKENS=4096         # stage 1
ANS_MAX_NEW_TOKENS=4096         # stage 2 (the model thinks again)
TEMPERATURE=0.6                 # Qwen3 thinking-mode recommendation
TOP_P=0.95
TOP_K=20
SEED=42                         # per request: sha256("42:<_id>:<cot|answer>") mod 2^31; server --seed 42
ENABLE_THINKING=true            # chat_template_kwargs.enable_thinking, both stages
N_PROC=4                        # concurrent questions (= --max-num-seqs)
DTYPE="${DTYPE:-bfloat16}"
GPU_MEM=0.88
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"   # hardware only (PCIe consumer GPUs); not part of the protocol
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
sampling          : temperature $TEMPERATURE, top_p $TOP_P, top_k $TOP_K, no repetition penalty, seed $SEED (per question/stage)
thinking          : enable_thinking=$ENABLE_THINKING in both stages; text before the first </think> is ignored by extract_answer
backend           : vLLM OpenAI server, dtype $DTYPE, --generation-config vllm, eager, max_num_seqs $N_PROC, max_num_batched_tokens 8192, TP $TP x PP $PP, GPUs $GPUS, gpu_mem $GPU_MEM
output            : $OUTPUT
EOF

start_server --dtype "$DTYPE" --max-model-len "$MAX_MODEL_LEN" --gpu-memory-utilization "$GPU_MEM" \
  --generation-config vllm --seed "$SEED" --max-num-seqs "$N_PROC" --max-num-batched-tokens 8192 \
  --enforce-eager --disable-custom-all-reduce

run_pred "${NAME}_cot.jsonl" \
  --model "$NAME" --model_path "$MODEL" --tokenizer "$PARENT" --cot \
  --max_len "$MAX_INPUT_TOKENS" \
  --cot_max_new_tokens "$COT_MAX_NEW_TOKENS" --ans_max_new_tokens "$ANS_MAX_NEW_TOKENS" \
  --temperature "$TEMPERATURE" --top_p "$TOP_P" --top_k "$TOP_K" \
  --enable_thinking "$ENABLE_THINKING" --seed "$SEED" \
  --dataset_revision "$DATASET_REVISION" --n_proc "$N_PROC"
