#!/usr/bin/env bash
# Shared runner for the OmniZipEval LiveCodeBench scripts. Do not call directly:
# qwen25_instruct.sh, deepseek_r1_distill.sh and qwen3.sh set the protocol
# constants below (readonly) and then source this file.
#
# Required (set by the family script):
#   PROTOCOL PARENTS DEFAULT_MODEL
#   SCENARIO RELEASE_VERSION DATASET_REVISION EXPECTED_PROBLEMS
#   N TEMPERATURE TOP_P MAX_TOKENS STOP MAX_MODEL_LEN TIMEOUT PROMPT_STYLE
#
# User inputs (environment variables):
#   MODEL      HF id or local directory of the checkpoint to evaluate
#              (dense or compressed). Default: DEFAULT_MODEL (the dense model).
#   PARENT     Dense original model the checkpoint was made from; must be one of
#              PARENTS. It selects the LiveCodeBench prompt style and output name.
#              Default: MODEL, when MODEL itself is one of PARENTS.
#   TOKENIZER  Tokenizer HF id or directory. Default: PARENT.
#   GPUS       Comma-separated device indices (default 0). All are used by one
#              vLLM engine: tensor_parallel x pipeline_parallel = number of GPUs.
#   TP_SIZE    Optional tensor-parallel size (must divide the GPU count).
#   DTYPE      bfloat16 (default) or float16 (for quantized checkpoints whose
#              vLLM kernel does not run in bfloat16; see README.md).
#   OUTPUT     Run directory (default runs/<protocol>/<model name>). Must not
#              already contain a finished run.
#   PYTHON     Python with lcb_runner's dependencies (default: python).
#   GPU_MEMORY_UTILIZATION (default 0.90), NUM_PROCESS_EVALUATE (default 12):
#              hardware only.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SCRIPT_DIR="$ROOT/scripts/omnizipeval"

MODEL="${MODEL:-$DEFAULT_MODEL}"
if [[ -z "${PARENT:-}" ]]; then
  for p in "${PARENTS[@]}"; do
    [[ "$MODEL" == "$p" ]] && PARENT="$p"
  done
fi
if [[ -z "${PARENT:-}" ]]; then
  echo "Error: MODEL=$MODEL is not a dense model of this protocol; set PARENT to one of:" >&2
  printf '  %s\n' "${PARENTS[@]}" >&2
  exit 1
fi
ok=0
for p in "${PARENTS[@]}"; do [[ "$PARENT" == "$p" ]] && ok=1; done
if (( ! ok )); then
  echo "Error: PARENT=$PARENT is not part of protocol $PROTOCOL; expected one of:" >&2
  printf '  %s\n' "${PARENTS[@]}" >&2
  exit 1
fi
TOKENIZER="${TOKENIZER:-$PARENT}"

# A model given as an existing directory is used as is; anything else is an HF id.
if [[ -d "$MODEL" ]]; then MODEL="$(cd "$MODEL" && pwd)"; fi
if [[ -d "$TOKENIZER" ]]; then TOKENIZER="$(cd "$TOKENIZER" && pwd)"; fi
if [[ "$MODEL" == /* && ! -d "$MODEL" ]]; then
  echo "Error: MODEL directory does not exist: $MODEL" >&2
  exit 1
fi

DTYPE="${DTYPE:-bfloat16}"
case "$DTYPE" in
  bfloat16|float16) ;;
  *) echo "Error: DTYPE must be bfloat16 or float16 (got $DTYPE)" >&2; exit 1 ;;
esac

GPUS="${GPUS:-0}"
IFS=',' read -r -a GPU_ARR <<< "$GPUS"
NUM_GPUS="${#GPU_ARR[@]}"
(( NUM_GPUS >= 1 )) || { echo "Error: no GPUs in GPUS=$GPUS" >&2; exit 1; }
if [[ -n "${TP_SIZE:-}" ]]; then
  (( NUM_GPUS % TP_SIZE == 0 )) || { echo "Error: TP_SIZE=$TP_SIZE does not divide $NUM_GPUS GPUs" >&2; exit 1; }
elif (( NUM_GPUS % 4 == 0 )); then
  TP_SIZE=4
elif (( NUM_GPUS % 2 == 0 )); then
  TP_SIZE=2
else
  TP_SIZE=1
fi
PP_SIZE=$(( NUM_GPUS / TP_SIZE ))
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
NUM_PROCESS_EVALUATE="${NUM_PROCESS_EVALUATE:-12}"
PYTHON="${PYTHON:-python}"

model_name="$(basename "$MODEL")"
OUTPUT="${OUTPUT:-$ROOT/runs/$PROTOCOL/$model_name}"
mkdir -p "$OUTPUT"
OUTPUT="$(cd "$OUTPUT" && pwd)"

# lcb_runner writes output/<model_repr>/... relative to the working directory;
# model_repr is the PARENT's name, so every checkpoint needs its own OUTPUT.
REPR="$(cd "$ROOT" && "$PYTHON" -c "from lcb_runner.lm_styles import LanguageModelStore as S; m=S['$PARENT']; print(m.model_repr); assert m.model_style.name == '$PROMPT_STYLE', m.model_style")"
EVAL_ALL="$OUTPUT/output/$REPR/Scenario.${SCENARIO}_${N}_${TEMPERATURE}_eval_all.json"
if [[ -e "$EVAL_ALL" ]]; then
  echo "Error: $EVAL_ALL already exists; choose a new OUTPUT." >&2
  exit 1
fi

"$PYTHON" -c "import vllm" 2>/dev/null || {
  echo "Error: vllm is not importable by $PYTHON. Install this folder first: pip install -e $ROOT" >&2
  exit 1
}

SETTINGS="$OUTPUT/settings.txt"
{
  echo "protocol:          $PROTOCOL"
  echo "lcb_runner:        LiveCodeBench 28fef95 + OmniZipEval modifications (see scripts/omnizipeval/README.md)"
  echo "model:             $MODEL"
  echo "parent:            $PARENT (lm_styles model_repr=$REPR)"
  echo "prompt style:      $PROMPT_STYLE"
  echo "tokenizer:         $TOKENIZER"
  echo "dataset:           livecodebench/code_generation_lite @ $DATASET_REVISION"
  echo "release_version:   $RELEASE_VERSION (no start/end date filter; $EXPECTED_PROBLEMS problems)"
  echo "scenario:          $SCENARIO"
  echo "n:                 $N"
  echo "temperature:       $TEMPERATURE"
  echo "top_p:             $TOP_P"
  echo "max_tokens:        $MAX_TOKENS"
  echo "stop:              $STOP"
  echo "max_model_len:     $MAX_MODEL_LEN"
  echo "dtype:             $DTYPE"
  echo "vllm:              enforce_eager, no prefix caching, trust_remote_code"
  echo "generation cache:  off"
  echo "evaluate:          --evaluate, timeout=${TIMEOUT}s, num_process_evaluate=$NUM_PROCESS_EVALUATE"
  echo "gpus:              $GPUS (tensor_parallel=$TP_SIZE x pipeline_parallel=$PP_SIZE)"
  echo "gpu_memory_util:   $GPU_MEMORY_UTILIZATION"
  echo "output:            $OUTPUT"
} | tee "$SETTINGS"

export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export LCB_DATASET_REVISION="$DATASET_REVISION"
cd "$OUTPUT"

set +e
CUDA_VISIBLE_DEVICES="$GPUS" "$PYTHON" -u -m lcb_runner.runner.main \
  --model "$PARENT" \
  --local_model_path "$MODEL" \
  --tokenizer "$TOKENIZER" \
  --trust_remote_code \
  --scenario "$SCENARIO" \
  --release_version "$RELEASE_VERSION" \
  --n "$N" \
  --temperature "$TEMPERATURE" \
  --top_p "$TOP_P" \
  --max_tokens "$MAX_TOKENS" \
  --stop "$STOP" \
  --max_model_len "$MAX_MODEL_LEN" \
  --dtype "$DTYPE" \
  --tensor_parallel_size "$TP_SIZE" \
  --pipeline_parallel_size "$PP_SIZE" \
  --gpu_memory_utilization "$GPU_MEMORY_UTILIZATION" \
  --num_process_evaluate "$NUM_PROCESS_EVALUATE" \
  --timeout "$TIMEOUT" \
  --evaluate \
  2>&1 | tee "$OUTPUT/run.log"
status=${PIPESTATUS[0]}
set -e
if (( status != 0 )); then
  echo "Error: lcb_runner exited with $status (see $OUTPUT/run.log)" >&2
  exit "$status"
fi

"$PYTHON" "$SCRIPT_DIR/score.py" "$EVAL_ALL" \
  --expected-problems "$EXPECTED_PROBLEMS" --n "$N" --out "$OUTPUT/score.json"
