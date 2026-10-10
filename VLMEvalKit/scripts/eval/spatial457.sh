#!/usr/bin/env bash
# Spatial457: 6D spatial reasoning (5 levels, 7 sub-tasks), 23,752 questions on synthetic 3D scenes.
# VLMEvalKit dataset: Spatial457 (class Spatial457; VLMEvalKit checks the TSV md5). Scoring: the dataset's
# built-in matcher (ROBUST mode). No LLM judge.
#
# Usage (from anywhere):
#   FAMILY=qwen25_vl GPUS=0,1 bash scripts/eval/spatial457.sh                  # dense Qwen2.5-VL-3B
#   FAMILY=llava_next MODEL=<HF id or local dir> NAME=<run name> GPUS=0,1 bash scripts/eval/spatial457.sh
# FAMILY = qwen25_vl | llava_next | llava_v15. All generation
# settings are fixed below per FAMILY; MODEL (dense or compressed checkpoint) only changes the weights.
# Extra arguments go to run.py. See EVAL.md for the settings tables.
set -euo pipefail
KIT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$KIT"

DATASET=Spatial457
DATA_CLASS=Spatial457
FAMILY="${FAMILY:-qwen25_vl}"
case "$FAMILY" in
  qwen25_vl)    DENSE=Qwen/Qwen2.5-VL-3B-Instruct ;;
  llava_next)   DENSE=llava-hf/llava-v1.6-vicuna-7b-hf ;;
  llava_v15)    DENSE=llava-hf/llava-1.5-7b-hf ;;
  *) echo "FAMILY must be qwen25_vl, llava_next or llava_v15 for Spatial457" >&2; exit 2 ;;
esac
MODEL="${MODEL:-$DENSE}"
PROCESSOR="${PROCESSOR:-$DENSE}"   # LLaVA families: processor of the dense parent
NAME="${NAME:-$(basename "$MODEL")}"
GPUS="${GPUS:-0}"
WORK_DIR="${WORK_DIR:-outputs/spatial457}"
MASTER_PORT="${MASTER_PORT:-29500}"
PYTHON="${PYTHON:-python}"
IFS=',' read -r -a GPU_ARR <<< "$GPUS"
NGPU="${#GPU_ARR[@]}"
NPROC="${NPROC:-$NGPU}"            # transformers: model replicas; GPUs are split evenly between them

case "$FAMILY" in
  qwen25_vl)
    # vlmeval/config.py "Qwen2.5-VL-*-Instruct" registration + Qwen2VLChat default decoding
    # (use_custom_prompt=false: the dataset's own prompt).
    LAUNCH=torchrun; BACKEND="transformers (BF16), one replica per process"
    MODEL_JSON=$(cat <<EOF
{"class": "Qwen2VLChat", "model_path": "$MODEL",
 "min_pixels": 1003520, "max_pixels": 12845056,
 "max_new_tokens": 2048, "do_sample": true, "top_k": 1, "top_p": 0.001, "temperature": 0.01,
 "repetition_penalty": 1.0, "use_custom_prompt": false, "system_prompt": null, "use_vllm": false}
EOF
) ;;
  llava_next|llava_v15)
    # LLaVA_Next defaults, as in the vlmeval/config.py "llava_next_vicuna_7b" registration.
    LAUNCH=torchrun; BACKEND="transformers (FP16), one replica per process"
    MODEL_JSON=$(cat <<EOF
{"class": "LLaVA_Next", "model_path": "$MODEL", "processor_path": "$PROCESSOR",
 "max_new_tokens": 2048, "do_sample": false, "num_beams": 1}
EOF
) ;;
esac
JUDGE_ARGS=(--judge exact_matching)
REPORT="'score' (x100) in <NAME>_Spatial457_score.json; per level: L*_score"

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

export CUDA_VISIBLE_DEVICES="$GPUS" TOKENIZERS_PARALLELISM=false
unset LOCAL_LLM
RUN=("$PYTHON" -m torch.distributed.run --nproc-per-node="$NPROC" --master_port="$MASTER_PORT" run.py)
CMD=("${RUN[@]}" --config "$CFG" --work-dir "$WORK_DIR" --verbose "${JUDGE_ARGS[@]}" "$@")
echo "command: ${CMD[*]}"
[[ "${DRY_RUN:-0}" == 1 ]] && exit 0   # DRY_RUN=1: write and print the config only
"${CMD[@]}"
