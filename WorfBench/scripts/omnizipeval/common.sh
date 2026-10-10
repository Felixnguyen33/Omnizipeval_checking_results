# Shared launcher for the run_worfbench*.sh configurations. Source it, do not run it.
# The calling script hard-codes MAX_NEW_TOKENS, MAX_MODEL_LEN and FALLBACK_MAX_MODEL_LEN;
# everything else below is fixed for every OmniZipEval WorfBench run.

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORFBENCH_ROOT="$(cd "$HERE/../.." && pwd)"
PY="${PYTHON:-python}"

TASKS=(alfworld intercodesql lumos os seal_tools toolalpaca toolbench webshop)
TEMPERATURE=0.0
TOP_P=0.95
SEED=0
NODE_PARSER=optional_space
SCORER=sentence-transformers/all-mpnet-base-v2
SCORER_REVISION=e8c3b32edf5434bc2275fc9bab85f82640a19130

echo_settings() {
  echo "=== WorfBench (OmniZipEval protocol): $CONFIG"
  echo "model              : $MODEL"
  local tok="${TOKENIZER:-}"
  [[ -n "$tok" ]] || tok="(the model's own)"
  echo "tokenizer          : $tok - its chat template via vLLM LLM.chat, add_generation_prompt=True"
  echo "gpus               : $GPUS (one data-parallel vLLM worker per GPU, tensor_parallel_size=1)"
  echo "output             : $OUTPUT"
  echo "tasks              : ${TASKS[*]} (full gold_traj/<task>/graph_eval.json, equal-weight mean)"
  echo "gold sizes         : alfworld 312, intercodesql 500, lumos 489, os 20, seal_tools 223, toolalpaca 93, toolbench 114, webshop 133"
  echo "shots              : two-shot (prompts/eval_prompt.py two_shot_example) except intercodesql, seal_tools: zero-shot"
  echo "temperature / top_p: $TEMPERATURE / $TOP_P (greedy; top_k, repetition_penalty at vLLM defaults; seed $SEED)"
  echo "max_new_tokens     : $MAX_NEW_TOKENS"
  echo "max_model_len      : $MAX_MODEL_LEN (retry with ${FALLBACK_MAX_MODEL_LEN} + enforce_eager if the load fails; 0 = no retry)"
  echo "dtype              : auto, gpu_memory_utilization 0.90, trust_remote_code"
  echo "node parser        : $NODE_PARSER (regex \\d+[:.]\\s*(.+))"
  echo "scorer             : $SCORER @ $SCORER_REVISION, node eval (upstream t_eval_nodes, threshold 0.6)"
  echo "python             : $PY"
}

run_config() {
  local common=(
    --model "$MODEL" --output "$OUTPUT" --tasks "${TASKS[@]}"
    --temperature "$TEMPERATURE" --top-p "$TOP_P" --seed "$SEED"
    --max-new-tokens "$MAX_NEW_TOKENS" --max-model-len "$MAX_MODEL_LEN"
    --fallback-max-model-len "$FALLBACK_MAX_MODEL_LEN"
    --node-parser "$NODE_PARSER" --scorer "$SCORER" --scorer-revision "$SCORER_REVISION"
  )
  [[ -n "${TOKENIZER:-}" ]] && common+=(--tokenizer "$TOKENIZER")
  local gpu_list n i pids=() fail=0
  IFS=, read -ra gpu_list <<< "$GPUS"
  n=${#gpu_list[@]}
  mkdir -p "$OUTPUT/logs"
  if (( n == 1 )); then
    CUDA_VISIBLE_DEVICES="$GPUS" "$PY" -u "$HERE/generate.py" "${common[@]}" 2>&1 | tee -a "$OUTPUT/logs/run.log"
    return "${PIPESTATUS[0]}"
  fi
  for ((i = 0; i < n; i++)); do
    CUDA_VISIBLE_DEVICES="${gpu_list[$i]}" "$PY" -u "$HERE/generate.py" "${common[@]}" \
      --num-shards "$n" --shard-id "$i" --generate-only > "$OUTPUT/logs/shard$i.log" 2>&1 &
    pids+=($!)
    echo "shard $i on GPU ${gpu_list[$i]}: pid ${pids[-1]}, log $OUTPUT/logs/shard$i.log"
  done
  for i in "${pids[@]}"; do wait "$i" || fail=1; done
  (( fail == 0 )) || { echo "a generation shard failed; not scoring" >&2; return 1; }
  CUDA_VISIBLE_DEVICES="${gpu_list[0]}" "$PY" -u "$HERE/generate.py" "${common[@]}" \
    --merge-shards "$n" 2>&1 | tee -a "$OUTPUT/logs/score.log"
  return "${PIPESTATUS[0]}"
}
