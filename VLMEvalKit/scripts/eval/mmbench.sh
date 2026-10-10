#!/usr/bin/env bash
# MMBench: multiple-choice perception and reasoning, circular evaluation (a question counts only if all of
# its option rotations are answered correctly).
# VLMEvalKit dataset (version) per family:
#   qwen25_vl, llava_next, llava_v15 -> MMBench (v1.0 English, dev + test rows in one TSV);
#                                      report the "test" row (1,784 questions, 6,666 prompts).
#   qwen3_vl_moe                     -> MMBench_DEV_EN_V11 (v1.1 English dev, 1,292 questions, 4,876 prompts).
# Class ImageMCQDataset; VLMEvalKit checks the TSV md5. Answer matching: rule-based (exact_matching).
#
# Usage (from anywhere):
#   FAMILY=qwen25_vl GPUS=0,1 bash scripts/eval/mmbench.sh                     # dense Qwen2.5-VL-3B
#   FAMILY=qwen25_vl MODEL=<HF id or local dir> NAME=<run name> GPUS=0,1 bash scripts/eval/mmbench.sh
#   FAMILY=qwen25_vl MODEL=Qwen/Qwen2.5-VL-32B-Instruct GPUS=0,1,2,3 NPROC=1 bash scripts/eval/mmbench.sh
#   FAMILY=qwen3_vl_moe GPUS=0,1,2,3,4,5,6 bash scripts/eval/mmbench.sh
# FAMILY = qwen25_vl | llava_next | llava_v15 | qwen3_vl_moe. All generation settings are fixed below per
# FAMILY; MODEL (dense or compressed checkpoint) only changes the weights. Extra arguments go to run.py.
# See EVAL.md for the settings tables.
set -euo pipefail
KIT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$KIT"

DATA_CLASS=ImageMCQDataset
FAMILY="${FAMILY:-qwen25_vl}"
case "$FAMILY" in
  qwen25_vl)    DENSE=Qwen/Qwen2.5-VL-3B-Instruct;      DATASET=MMBench ;;
  llava_next)   DENSE=llava-hf/llava-v1.6-vicuna-7b-hf; DATASET=MMBench ;;
  llava_v15)    DENSE=llava-hf/llava-1.5-7b-hf;         DATASET=MMBench ;;
  qwen3_vl_moe) DENSE=Qwen/Qwen3-VL-30B-A3B-Instruct;   DATASET=MMBench_DEV_EN_V11 ;;
  *) echo "FAMILY must be qwen25_vl, llava_next, llava_v15 or qwen3_vl_moe" >&2; exit 2 ;;
esac
MODEL="${MODEL:-$DENSE}"
PROCESSOR="${PROCESSOR:-$DENSE}"   # LLaVA families: processor of the dense parent
NAME="${NAME:-$(basename "$MODEL")}"
GPUS="${GPUS:-0}"
WORK_DIR="${WORK_DIR:-outputs/mmbench}"
MASTER_PORT="${MASTER_PORT:-29500}"
PYTHON="${PYTHON:-python}"
IFS=',' read -r -a GPU_ARR <<< "$GPUS"
NGPU="${#GPU_ARR[@]}"
NPROC="${NPROC:-$NGPU}"            # transformers: model replicas; GPUs are split evenly between them

case "$FAMILY" in
  qwen25_vl)
    # vlmeval/config.py "Qwen2.5-VL-*-Instruct" registration + Qwen2VLChat default decoding.
    LAUNCH=torchrun; BACKEND="transformers (BF16), one replica per process"
    MODEL_JSON=$(cat <<EOF
{"class": "Qwen2VLChat", "model_path": "$MODEL",
 "min_pixels": 1003520, "max_pixels": 12845056,
 "max_new_tokens": 2048, "do_sample": true, "top_k": 1, "top_p": 0.001, "temperature": 0.01,
 "repetition_penalty": 1.0, "use_custom_prompt": false, "system_prompt": null, "use_vllm": false}
EOF
)
    REPORT="row 'test', column Overall of <NAME>_MMBench_acc.csv" ;;
  llava_next|llava_v15)
    # LLaVA_Next defaults, as in the vlmeval/config.py "llava_next_vicuna_7b" registration
    # (LLaVA_Next builds its own MCQ prompt: "Answer with the option's letter from the given choices directly.").
    LAUNCH=torchrun; BACKEND="transformers (FP16), one replica per process"
    MODEL_JSON=$(cat <<EOF
{"class": "LLaVA_Next", "model_path": "$MODEL", "processor_path": "$PROCESSOR",
 "max_new_tokens": 2048, "do_sample": false, "num_beams": 1}
EOF
)
    REPORT="row 'test', column Overall of <NAME>_MMBench_acc.csv" ;;
  qwen3_vl_moe)
    # Same settings as the Qwen3-VL suite (scripts/run_qwen3_vl_suite.py): greedy, 128 tokens,
    # Qwen3-VL MCQ prompt, 4..16384 visual tokens, 32k context, vLLM pipeline parallel over GPUS.
    LAUNCH=python; BACKEND="vLLM (BF16), TP=1, PP=$NGPU, one process"
    MODEL_JSON=$(cat <<EOF
{"class": "Qwen3VLChat", "model_path": "$MODEL",
 "min_pixels": 4096, "max_pixels": 16777216,
 "max_new_tokens": 128, "temperature": 0.0, "top_p": 1.0, "top_k": -1,
 "repetition_penalty": 1.0, "presence_penalty": 0.0,
 "use_custom_prompt": true, "system_prompt": null, "use_vllm": true, "tensor_parallel_size": 1,
 "vllm_kwargs": {"dtype": "bfloat16", "pipeline_parallel_size": $NGPU, "distributed_executor_backend": "mp",
   "enable_expert_parallel": false, "max_model_len": 32768, "max_num_batched_tokens": 8192,
   "enforce_eager": true, "disable_custom_all_reduce": true, "gpu_memory_utilization": 0.85, "seed": 0,
   "limit_mm_per_prompt": {"image": 1, "video": 0},
   "mm_processor_kwargs": {"min_pixels": 4096, "max_pixels": 16777216},
   "hf_overrides": {"text_config": {"architectures": ["Qwen3MoeForCausalLM"]}}}}
EOF
)
    REPORT="row 'dev', column Overall of <NAME>_MMBench_DEV_EN_V11_acc.csv" ;;
esac
JUDGE_ARGS=(--judge exact_matching)

mkdir -p "$WORK_DIR"
CFG="$WORK_DIR/${NAME}_${DATASET}_config.json"
export _MODEL_JSON="$MODEL_JSON"
{
  echo "dataset=$DATASET (class $DATA_CLASS)  family=$FAMILY  model=$MODEL  name=$NAME"
  echo "backend=$BACKEND  gpus=$GPUS  nproc=$([[ $LAUNCH == python ]] && echo 1 || echo "$NPROC")"
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
if [[ "$LAUNCH" == python ]]; then
  RUN=("$PYTHON" run.py)
else
  RUN=("$PYTHON" -m torch.distributed.run --nproc-per-node="$NPROC" --master_port="$MASTER_PORT" run.py)
fi
CMD=("${RUN[@]}" --config "$CFG" --work-dir "$WORK_DIR" --verbose "${JUDGE_ARGS[@]}" "$@")
echo "command: ${CMD[*]}"
[[ "${DRY_RUN:-0}" == 1 ]] && exit 0   # DRY_RUN=1: write and print the config only
"${CMD[@]}"
