#!/usr/bin/env bash
# Rescore a finished run_worfbench*.sh output with upstream WorfBench's own node parser
# (regex \d+[:.] (.+), a space required after "1:" / "1."). No generation, no model load.
# Writes OUTPUT/eval_strict_space/ and OUTPUT/summary_strict_space.json next to the default
# optional_space scores (eval_optional_space/, summary_optional_space.json). Optional
# parser-sensitivity check; the OmniZipEval metric uses optional_space.
#
# Usage: OUTPUT=<run dir> [GPUS=0] [PYTHON=python] bash scripts/omnizipeval/rescore_strict_parser.sh
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PYTHON:-python}"
: "${OUTPUT:?set OUTPUT to a finished run directory}"
GPUS="${GPUS:-0}"

NODE_PARSER=strict_space
SCORER=sentence-transformers/all-mpnet-base-v2
SCORER_REVISION=e8c3b32edf5434bc2275fc9bab85f82640a19130

echo "=== WorfBench rescore with the upstream parser"
echo "output     : $OUTPUT (tasks, limit and generation settings read from its settings.json)"
echo "node parser: $NODE_PARSER (regex \\d+[:.] (.+))"
echo "scorer     : $SCORER @ $SCORER_REVISION on GPU ${GPUS%%,*}"
CUDA_VISIBLE_DEVICES="${GPUS%%,*}" "$PY" -u "$HERE/generate.py" --score-only --output "$OUTPUT" \
  --node-parser "$NODE_PARSER" --scorer "$SCORER" --scorer-revision "$SCORER_REVISION"
