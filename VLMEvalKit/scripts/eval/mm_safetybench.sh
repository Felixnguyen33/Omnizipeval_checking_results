#!/usr/bin/env bash
# MM-SafetyBench: 13 harmful-content scenarios. Each answer is judged safe / unsafe by an LLM judge;
# safety rate = 100 - attack success rate.
# VLMEvalKit dataset: MMSafetyBench (class MMSafetyBenchDataset; VLMEvalKit's packaging: 1,680 items, one SD
# image + the original question each; VLMEvalKit checks the TSV md5). Judge prompts and label parsing are
# VLMEvalKit's own (mmsafetybench.py).
#
# Judge: a local OpenAI-compatible server, no API key: Qwen/Qwen3-30B-A3B
# (revision ad44e777bcd18fa416d9da3bd8f70d33ebb85d39), BF16, thinking disabled, greedy, 128 tokens, e.g.
#   vllm serve Qwen/Qwen3-30B-A3B --revision ad44e777bcd18fa416d9da3bd8f70d33ebb85d39 --dtype bfloat16 \
#     --max-model-len 8192 --seed 0 --port 8000
#   JUDGE_BASE_URL=http://127.0.0.1:8000/v1 JUDGE_MODEL=Qwen/Qwen3-30B-A3B FAMILY=... bash scripts/eval/mm_safetybench.sh
# Without JUDGE_BASE_URL the script only runs inference (pass --mode infer explicitly to confirm).
#
# FAMILY = qwen25_vl | qwen3_vl_moe. All generation settings are fixed below per FAMILY; MODEL (dense or
# compressed checkpoint) only changes the weights. Extra arguments go to run.py. See EVAL.md for the
# settings tables.
set -euo pipefail
KIT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$KIT"

DATASET=MMSafetyBench
DATA_CLASS=MMSafetyBenchDataset
FAMILY="${FAMILY:-qwen25_vl}"
case "$FAMILY" in
  qwen25_vl)    DENSE=Qwen/Qwen2.5-VL-7B-Instruct ;;
  qwen3_vl_moe) DENSE=Qwen/Qwen3-VL-30B-A3B-Instruct ;;
  *) echo "FAMILY must be qwen25_vl or qwen3_vl_moe for MM-SafetyBench" >&2; exit 2 ;;
esac
MODEL="${MODEL:-$DENSE}"
NAME="${NAME:-$(basename "$MODEL")}"
GPUS="${GPUS:-0}"
WORK_DIR="${WORK_DIR:-outputs/mm_safetybench}"
MASTER_PORT="${MASTER_PORT:-29500}"
PYTHON="${PYTHON:-python}"
IFS=',' read -r -a GPU_ARR <<< "$GPUS"
NGPU="${#GPU_ARR[@]}"
NPROC="${NPROC:-$NGPU}"            # transformers: model replicas; GPUs are split evenly between them

case "$FAMILY" in
  qwen25_vl)
    # vlmeval/config.py "Qwen2.5-VL-*-Instruct" registration + Qwen2VLChat default decoding,
    # 2,048 new tokens; dataset prompt, no system prompt.
    LAUNCH=torchrun; BACKEND="transformers (BF16), one replica per process"
    MODEL_JSON=$(cat <<EOF
{"class": "Qwen2VLChat", "model_path": "$MODEL",
 "min_pixels": 1003520, "max_pixels": 12845056,
 "max_new_tokens": 2048, "do_sample": true, "top_k": 1, "top_p": 0.001, "temperature": 0.01,
 "repetition_penalty": 1.0, "use_custom_prompt": false, "system_prompt": null, "use_vllm": false}
EOF
) ;;
  qwen3_vl_moe)
    # Same settings as the Qwen3-VL suite (scripts/run_qwen3_vl_suite.py): greedy, 2,048 tokens, the
    # dataset's own prompt (use_custom_prompt=false), 4..16384 visual tokens, 32k context, vLLM PP over GPUS.
    LAUNCH=python; BACKEND="vLLM (BF16), TP=1, PP=$NGPU, one process"
    MODEL_JSON=$(cat <<EOF
{"class": "Qwen3VLChat", "model_path": "$MODEL",
 "min_pixels": 4096, "max_pixels": 16777216,
 "max_new_tokens": 2048, "temperature": 0.0, "top_p": 1.0, "top_k": -1,
 "repetition_penalty": 1.0, "presence_penalty": 0.0,
 "use_custom_prompt": false, "system_prompt": null, "use_vllm": true, "tensor_parallel_size": 1,
 "vllm_kwargs": {"dtype": "bfloat16", "pipeline_parallel_size": $NGPU, "distributed_executor_backend": "mp",
   "enable_expert_parallel": false, "max_model_len": 32768, "max_num_batched_tokens": 8192,
   "enforce_eager": true, "disable_custom_all_reduce": true, "gpu_memory_utilization": 0.85, "seed": 0,
   "limit_mm_per_prompt": {"image": 1, "video": 0},
   "mm_processor_kwargs": {"min_pixels": 4096, "max_pixels": 16777216},
   "hf_overrides": {"text_config": {"architectures": ["Qwen3MoeForCausalLM"]}}}}
EOF
) ;;
esac

# Judge: local OpenAI-compatible server only. Judge decoding is fixed: greedy, 128 tokens, seed 0, no thinking.
JUDGE_MODEL="${JUDGE_MODEL:-Qwen/Qwen3-30B-A3B}"
JUDGE_DECODING='{"temperature": 0, "max_tokens": 128, "seed": 0, "chat_template_kwargs": {"enable_thinking": false}}'
if [[ -n "${JUDGE_BASE_URL:-}" ]]; then
  JUDGE_ARGS=(--judge "$JUDGE_MODEL" --judge-base-url "$JUDGE_BASE_URL" --judge-key "${JUDGE_KEY:-EMPTY}"
              --judge-args "$JUDGE_DECODING")
  JUDGE_DESC="$JUDGE_MODEL at $JUDGE_BASE_URL, $JUDGE_DECODING"
else
  if [[ " $* " != *" --mode infer "* && "${DRY_RUN:-0}" != 1 ]]; then
    echo "Set JUDGE_BASE_URL (local judge server, see the header) or pass --mode infer for inference only." >&2
    exit 2
  fi
  JUDGE_ARGS=(--judge exact_matching)
  JUDGE_DESC="none (inference only)"
fi
REPORT="row Overall, column safety_rate (x100) of <NAME>_MMSafetyBench_score.csv; require unknown == 0 (VLMEvalKit counts unknown as safe)"

mkdir -p "$WORK_DIR"
CFG="$WORK_DIR/${NAME}_${DATASET}_config.json"
export _MODEL_JSON="$MODEL_JSON"
{
  echo "dataset=$DATASET (class $DATA_CLASS)  family=$FAMILY  model=$MODEL  name=$NAME"
  echo "backend=$BACKEND  gpus=$GPUS  nproc=$([[ $LAUNCH == python ]] && echo 1 || echo "$NPROC")"
  echo "judge=$JUDGE_DESC"
  echo "report=$REPORT"
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
