# Shared helpers for the OmniZipEval LongBench v2 scripts. Sourced, not run.
#
# Every protocol setting lives in the calling script. This file only starts the
# vLLM OpenAI-compatible server, runs pred.py (resuming until all questions are
# saved) and scores with result.py.

LB_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${PYTHON:-python}"

# LongBench v2 data: THUDM/LongBench-v2, split "train" (the only split), 503 questions.
DATASET_ID="THUDM/LongBench-v2"
DATASET_REVISION="2b48e494f2c7a2f0af81aae178e05c7e1dde0fe9"
DATASET_SIZE=503

API_KEY="token-abc123"   # local-only key shared by the server and pred.py
SERVER_PID=""

common_env() {
  # Runtime (non-protocol) settings shared by all scripts.
  MODEL="${MODEL:-$DEFAULT_MODEL}"
  PARENT="${PARENT:-$MODEL}"                       # tokenizer + chat template source
  NAME="${NAME:-$(basename "$MODEL")}"             # run name; names the prediction file
  GPUS="${GPUS:-$DEFAULT_GPUS}"
  IFS=',' read -r -a _gpu_list <<< "$GPUS"
  TP="${TP:-${#_gpu_list[@]}}"                      # tensor parallel (hardware placement)
  PP="${PP:-1}"                                     # pipeline parallel (hardware placement)
  OUTPUT="${OUTPUT:-$LB_ROOT/outputs/$SCRIPT_TAG/$NAME}"
  PRED_PASSES="${PRED_PASSES:-2}"                   # pred.py resumes missing ids on each pass
  mkdir -p "$OUTPUT/predictions"
  OUTPUT="$(cd "$OUTPUT" && pwd)"
}

find_free_port() {
  "$PYTHON" - <<'PY'
import socket
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind(("127.0.0.1", 0))
print(s.getsockname()[1])
s.close()
PY
}

stop_server() {
  if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "[server] stopping vLLM (pid $SERVER_PID)"
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
}

# start_server <extra vllm args...>
start_server() {
  PORT="${PORT:-$(find_free_port)}"
  trap stop_server EXIT
  local cmd=("$PYTHON" -m vllm.entrypoints.openai.api_server
             --model "$MODEL" --tokenizer "$PARENT" --served-model-name "$MODEL"
             --host 127.0.0.1 --port "$PORT" --api-key "$API_KEY"
             --tensor-parallel-size "$TP" --pipeline-parallel-size "$PP" "$@")
  echo "[server] CUDA_VISIBLE_DEVICES=$GPUS ${cmd[*]}" | tee "$OUTPUT/server_command.txt"
  CUDA_VISIBLE_DEVICES="$GPUS" "${cmd[@]}" > "$OUTPUT/vllm_server.log" 2>&1 &
  SERVER_PID=$!
  until curl -sf -H "Authorization: Bearer $API_KEY" "http://127.0.0.1:${PORT}/v1/models" >/dev/null; do
    sleep 5
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
      echo "[server] vLLM died; see $OUTPUT/vllm_server.log" >&2
      tail -n 40 "$OUTPUT/vllm_server.log" >&2 || true
      exit 1
    fi
  done
  echo "[server] ready on port $PORT"
  export LONGBENCH_API_BASE="http://127.0.0.1:${PORT}/v1"
  export LONGBENCH_API_KEY="$API_KEY"
}

count_ids() {
  [[ -f "$1" ]] || { echo 0; return; }
  "$PYTHON" -c 'import json,sys; print(len({json.loads(l)["_id"] for l in open(sys.argv[1], encoding="utf-8") if l.strip()}))' "$1"
}

# run_pred <prediction file name> <pred.py args...>
run_pred() {
  local pred_file="$OUTPUT/predictions/$1"; shift
  local pass
  for (( pass = 1; pass <= PRED_PASSES; pass++ )); do
    echo "[pred] pass $pass: $(count_ids "$pred_file")/$DATASET_SIZE saved"
    (cd "$LB_ROOT" && "$PYTHON" pred.py --save_dir "$OUTPUT/predictions" "$@") 2>&1 | tee -a "$OUTPUT/pred.log"
    [[ "$(count_ids "$pred_file")" -ge "$DATASET_SIZE" ]] && break
  done
  local n; n="$(count_ids "$pred_file")"
  if [[ "$n" -lt "$DATASET_SIZE" ]]; then
    echo "[pred] INCOMPLETE: $n/$DATASET_SIZE questions in $pred_file (failed requests are dropped by pred.py; run again to resume)" >&2
    exit 1
  fi
  (cd "$LB_ROOT" && "$PYTHON" result.py --results_dir "$OUTPUT/predictions" --output "$OUTPUT/result.txt" --expected "$DATASET_SIZE")
  echo "[done] predictions: $pred_file"
  echo "[done] scores (Overall Easy Hard Short Medium Long): $OUTPUT/result.txt"
}
