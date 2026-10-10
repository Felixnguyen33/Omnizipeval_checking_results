#!/usr/bin/env bash
# ScreenSpot v2: GUI element grounding, 1,272 instructions (Desktop 334, Mobile 501, Web 437).
# VLMEvalKit dataset: ScreenSpot_v2 (class ConcatDataset of ScreenSpot_v2_Desktop/_Mobile/_Web; VLMEvalKit
# checks each TSV md5). Scoring: point-in-box (ScreenSpot.evaluate_point). The ScreenSpot score is the mean
# of the Desktop / Mobile / Web Overall_Accuracy values.
#
# Prompt / protocol per family:
#   qwen25_vl, llava_*: VLMEvalKit's native ScreenSpot prompt (pyautogui.click). For Qwen2.5-VL the local
#                       computer_use adapter of this copy is switched off (qwen25_screenspot_adapter=false).
#   qwen3_vl_moe:       the Qwen3-VL suite (scripts/run_qwen3_vl_suite.py): {"point_2d": [x, y]} prompt
#                       on Qwen3's 0-1000 scale, converted to fractions for the same scorer. VLMEvalKit's
#                       Qwen3VLChat has no such prompt, so this family runs the suite, not run.py.
#
# Usage (from anywhere):
#   FAMILY=qwen25_vl GPUS=0,1 bash scripts/eval/screenspot_v2.sh                # dense Qwen2.5-VL-3B
#   FAMILY=qwen25_vl MODEL=<HF id or local dir> NAME=<run name> GPUS=0,1 bash scripts/eval/screenspot_v2.sh
#   FAMILY=qwen3_vl_moe GPUS=0,1,2,3,4,5,6 bash scripts/eval/screenspot_v2.sh
# FAMILY = qwen25_vl | llava_next | llava_v15 | qwen3_vl_moe. All generation settings are fixed below per
# FAMILY; MODEL (dense or compressed checkpoint) only changes the weights. Extra arguments go to run.py
# (or to the Qwen3-VL suite). See EVAL.md for the settings tables.
set -euo pipefail
KIT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$KIT"

DATASET=ScreenSpot_v2
DATA_CLASS=ConcatDataset
FAMILY="${FAMILY:-qwen25_vl}"
case "$FAMILY" in
  qwen25_vl)    DENSE=Qwen/Qwen2.5-VL-3B-Instruct ;;
  llava_next)   DENSE=llava-hf/llava-v1.6-vicuna-7b-hf ;;
  llava_v15)    DENSE=llava-hf/llava-1.5-7b-hf ;;
  qwen3_vl_moe) DENSE=Qwen/Qwen3-VL-30B-A3B-Instruct ;;
  *) echo "FAMILY must be qwen25_vl, llava_next, llava_v15 or qwen3_vl_moe" >&2; exit 2 ;;
esac
MODEL="${MODEL:-$DENSE}"
PROCESSOR="${PROCESSOR:-$DENSE}"   # LLaVA families: processor of the dense parent
NAME="${NAME:-$(basename "$MODEL")}"
GPUS="${GPUS:-0}"
WORK_DIR="${WORK_DIR:-outputs/screenspot_v2}"
MASTER_PORT="${MASTER_PORT:-29500}"
PYTHON="${PYTHON:-python}"
IFS=',' read -r -a GPU_ARR <<< "$GPUS"
NGPU="${#GPU_ARR[@]}"
NPROC="${NPROC:-$NGPU}"            # transformers: model replicas; GPUs are split evenly between them

case "$FAMILY" in
  qwen25_vl)
    # vlmeval/config.py "Qwen2.5-VL-*-Instruct" registration + Qwen2VLChat default decoding;
    # dataset (pyautogui) prompt, local computer_use adapter off.
    LAUNCH=torchrun; BACKEND="transformers (BF16), one replica per process"
    MODEL_JSON=$(cat <<EOF
{"class": "Qwen2VLChat", "model_path": "$MODEL",
 "min_pixels": 1003520, "max_pixels": 12845056,
 "max_new_tokens": 2048, "do_sample": true, "top_k": 1, "top_p": 0.001, "temperature": 0.01,
 "repetition_penalty": 1.0, "use_custom_prompt": false, "system_prompt": null, "use_vllm": false,
 "qwen25_screenspot_adapter": false}
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
  qwen3_vl_moe)
    # Not a run.py config: the settings of scripts/run_qwen3_vl_suite.py, recorded for the work dir.
    LAUNCH=qwen3_suite; BACKEND="vLLM (BF16), TP=1, PP=$NGPU, batch 16 (scripts/run_qwen3_vl_suite.py)"
    MODEL_JSON=$(cat <<EOF
{"runner": "scripts/run_qwen3_vl_suite.py infer", "model_path": "$MODEL",
 "prompt": "Locate the UI element described as: <instruction>\\nReturn the center of the element as JSON: {\\"point_2d\\": [x, y]}. Use integer coordinates normalized to the range 0 to 1000. Return only the JSON.",
 "min_pixels": 4096, "max_pixels": 16777216, "max_new_tokens": 128, "temperature": 0.0, "seed": 0,
 "dtype": "bfloat16", "tensor_parallel_size": 1, "pipeline_parallel_size": $NGPU, "max_model_len": 32768,
 "coordinates": "point_2d / 1000 -> fraction; unparsable answer -> miss (out-of-bounds point)"}
EOF
) ;;
esac
JUDGE_ARGS=(--judge exact_matching)
REPORT="Overall_Accuracy of <NAME>_ScreenSpot_v2_{Desktop,Mobile,Web}_score.json; ScreenSpot score = their mean"
[[ "$FAMILY" == qwen3_vl_moe ]] && REPORT="<WORK_DIR>/<NAME>/ScreenSpot_v2.metrics.json: {Desktop,Mobile,Web}:Overall_Accuracy (ScreenSpot score = their mean)"

mkdir -p "$WORK_DIR"
CFG="$WORK_DIR/${NAME}_${DATASET}_config.json"
export _MODEL_JSON="$MODEL_JSON"
{
  echo "dataset=$DATASET (class $DATA_CLASS)  family=$FAMILY  model=$MODEL  name=$NAME"
  echo "backend=$BACKEND  gpus=$GPUS  nproc=$([[ $LAUNCH == torchrun ]] && echo "$NPROC" || echo 1)"
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
if [[ "$LAUNCH" == qwen3_suite ]]; then
  # Writes <WORK_DIR>/<NAME>/ScreenSpot_v2.metrics.json (Desktop/Mobile/Web Overall_Accuracy + pooled).
  CMD=("$PYTHON" scripts/run_qwen3_vl_suite.py infer --model "$MODEL" --name "$NAME" --output "$WORK_DIR"
       --pp "$NGPU" --batch-size 16 --datasets ScreenSpot_v2 "$@")
else
  RUN=("$PYTHON" -m torch.distributed.run --nproc-per-node="$NPROC" --master_port="$MASTER_PORT" run.py)
  CMD=("${RUN[@]}" --config "$CFG" --work-dir "$WORK_DIR" --verbose "${JUDGE_ARGS[@]}" "$@")
fi
echo "command: ${CMD[*]}"
[[ "${DRY_RUN:-0}" == 1 ]] && exit 0   # DRY_RUN=1: write and print the config only
"${CMD[@]}"
