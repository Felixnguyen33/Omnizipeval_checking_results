#!/usr/bin/env bash
# WeMath (We-Math): visual math multiple-choice, 1,740 questions (525 problems, 1-3 steps).
# VLMEvalKit dataset: WeMath (class WeMath; VLMEvalKit checks the TSV md5). VLMEvalKit reports the WeMath
# Score (Strict / Loose) and four-dimension rates; answers are matched rule-based (exact_matching).
# OmniZipEval's WeMath metric is letter accuracy over the 1,740 questions, computed outside VLMEvalKit from
# the prediction file this script writes (see EVAL.md).
#
# Usage (from anywhere):
#   FAMILY=qwen25_vl GPUS=0,1,2,3 bash scripts/eval/wemath.sh                  # dense Qwen2.5-VL-3B, vLLM x4
#   FAMILY=qwen25_vl MODEL=<HF id or local dir> NAME=<run name> GPUS=0,1,2,3 bash scripts/eval/wemath.sh
#   FAMILY=llava_next GPUS=0,1 bash scripts/eval/wemath.sh
# FAMILY = qwen25_vl | llava_next | llava_v15. All generation
# settings are fixed below per FAMILY; MODEL (dense or compressed checkpoint) only changes the weights.
# Extra arguments go to run.py. See EVAL.md for the settings tables.
set -euo pipefail
KIT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$KIT"

DATASET=WeMath
DATA_CLASS=WeMath
FAMILY="${FAMILY:-qwen25_vl}"
case "$FAMILY" in
  qwen25_vl)    DENSE=Qwen/Qwen2.5-VL-3B-Instruct ;;
  llava_next)   DENSE=llava-hf/llava-v1.6-vicuna-7b-hf ;;
  llava_v15)    DENSE=llava-hf/llava-1.5-7b-hf ;;
  *) echo "FAMILY must be qwen25_vl, llava_next or llava_v15 for WeMath" >&2; exit 2 ;;
esac
MODEL="${MODEL:-$DENSE}"
PROCESSOR="${PROCESSOR:-$DENSE}"   # LLaVA families: processor of the dense parent
NAME="${NAME:-$(basename "$MODEL")}"
GPUS="${GPUS:-0}"
WORK_DIR="${WORK_DIR:-outputs/wemath}"
MASTER_PORT="${MASTER_PORT:-29500}"
PYTHON="${PYTHON:-python}"
IFS=',' read -r -a GPU_ARR <<< "$GPUS"
NGPU="${#GPU_ARR[@]}"
NPROC="${NPROC:-$NGPU}"            # model replicas; GPUs are split evenly between them

case "$FAMILY" in
  qwen25_vl)
    # vLLM, greedy, 16,000 new tokens,
    # 32,768-token context, BF16, seed 0, image bounds 1280..16384 x 28 x 28, and Qwen2VLChat's MCQ prompt
    # ("Hint/Question/Options ... Please select the correct answer from the options above.") =
    # use_custom_prompt=true.
    LAUNCH=torchrun; BACKEND="vLLM (BF16) inside Qwen2VLChat, TP=1, one engine per process"
    MODEL_JSON=$(cat <<EOF
{"class": "Qwen2VLChat", "model_path": "$MODEL",
 "min_pixels": 1003520, "max_pixels": 12845056,
 "max_new_tokens": 16000, "do_sample": false, "top_k": 1, "top_p": 0.001, "temperature": 0.01,
 "repetition_penalty": 1.0, "use_custom_prompt": true, "system_prompt": null, "use_vllm": true,
 "vllm_kwargs": {"dtype": "bfloat16", "seed": 0, "max_model_len": 32768, "enforce_eager": true,
   "gpu_memory_utilization": 0.85, "limit_mm_per_prompt": {"image": 1, "video": 0},
   "mm_processor_kwargs": {"min_pixels": 1003520, "max_pixels": 12845056}}}
EOF
) ;;
  llava_next|llava_v15)
    # LLaVA_Next, greedy, 16,000 new tokens.
    # LLaVA_Next uses its own MCQ prompt ("Answer with the option's letter from the given choices directly.").
    LAUNCH=torchrun; BACKEND="transformers (FP16), one replica per process"
    MODEL_JSON=$(cat <<EOF
{"class": "LLaVA_Next", "model_path": "$MODEL", "processor_path": "$PROCESSOR",
 "max_new_tokens": 16000, "do_sample": false, "num_beams": 1}
EOF
) ;;
esac
JUDGE_ARGS=(--judge exact_matching)
REPORT="Score (Strict) / Score (Loose) in <NAME>_WeMath_exact_matching_score.csv; letter accuracy: see EVAL.md"

mkdir -p "$WORK_DIR"
CFG="$WORK_DIR/${NAME}_${DATASET}_config.json"
export _MODEL_JSON="$MODEL_JSON"
{
  echo "dataset=$DATASET (class $DATA_CLASS)  family=$FAMILY  model=$MODEL  name=$NAME"
  echo "backend=$BACKEND  gpus=$GPUS  nproc=$NPROC"
  echo "judge=${JUDGE_ARGS[*]}  report=$REPORT"
  echo "config=$CFG"
  "$PYTHON" - "$CFG" "$NAME" "$DATASET" "$DATA_CLASS" <<'PY'
import json, os, sys
path, name, dataset, data_class = sys.argv[1:5]
cfg = {"model": {name: json.loads(os.environ["_MODEL_JSON"])},
       "data": {dataset: {"class": data_class, "dataset": dataset}}}
open(path, "w").write(json.dumps(cfg, indent=2) + "\n")
print(json.dumps(cfg, indent=2))
PY
} | tee "$WORK_DIR/${NAME}_${DATASET}_settings.txt"

export CUDA_VISIBLE_DEVICES="$GPUS" VLLM_WORKER_MULTIPROC_METHOD=spawn TOKENIZERS_PARALLELISM=false
unset LOCAL_LLM
RUN=("$PYTHON" -m torch.distributed.run --nproc-per-node="$NPROC" --master_port="$MASTER_PORT" run.py)
CMD=("${RUN[@]}" --config "$CFG" --work-dir "$WORK_DIR" --verbose "${JUDGE_ARGS[@]}" "$@")
echo "command: ${CMD[*]}"
[[ "${DRY_RUN:-0}" == 1 ]] && exit 0   # DRY_RUN=1: write and print the config only
"${CMD[@]}"
