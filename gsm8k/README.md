# GSM8K

`evaluate.py` evaluates one checkpoint on the GSM8K test split with lm-eval
(`task.yaml`, scorer in `utils.py`). `check_pair.py` checks that a dense run and
a compressed-variant run are comparable.

## Install

Python 3.10+ with a GPU build of PyTorch. From this folder:

```bash
python -m pip install -r requirements.txt   # lm-eval 0.4.12 from ../lm-evaluation-harness [hf]
```

## Run

```bash
# 1. dense parent (--model and --parent-tokenizer are the same)
python evaluate.py \
  --model /path/to/dense-parent \
  --parent-tokenizer /path/to/dense-parent \
  --output outputs/base

# 2. compressed variant, same dense parent, with what it should be
python evaluate.py \
  --model /path/to/compressed-checkpoint \
  --parent-tokenizer /path/to/dense-parent \
  --expect-sparsity 0.5 \
  --output outputs/variant
#   pruned 50%: --expect-sparsity 0.5   GPTQ/AWQ-Int4: --expect-bits 4   SmoothQuant W8A8: --expect-bits 8

# 3. pair check
python check_pair.py outputs/base outputs/variant
```

For Hub IDs, pass `--revision` and `--parent-revision` as full 40-character
commit SHAs. Use `--parallelize` (HF sharding), the batch size and the hardware
consistently within a pair. `--smoke N` evaluates only the first N questions and
is recorded in the protocol, so a smoke run never pairs with a full run. Existing
output directories are refused.

## Settings

| Setting | Value |
| --- | --- |
| Dataset | `openai/gsm8k`, config `main`, revision `740312add88f781978c0658806c59bc2815b9866` |
| Split | `test`, 1,319 questions |
| Shots | 4 fixed demonstrations (`utils.py`), given as separate chat turns |
| Prompt | `Question: {question}\nPlease reason step by step, and put your final answer within \boxed{}.` |
| Chat template | dense parent's tokenizer, `enable_thinking=False` where supported |
| Decoding | greedy (`do_sample=False`, temperature 0), stop on `Question:` / `<\|im_end\|>` |
| Budget | 4,096 generated tokens, 8,192-token context |
| Backend | lm-eval HF backend, BF16 weights, float32 softmax, batch size 1 by default |
| Seeds | 0 (random, numpy, torch, few-shot); no response or request cache |

## Metrics

- `exact_match`: the last complete `\boxed{...}` after `</think>` (when present)
  must equal the gold number. Accepted: signed integers/decimals, grouping
  commas, an optional dollar sign. Fractions, expressions, prose-only answers,
  incomplete boxes and unfinished `<think>` reasoning score 0. There is no
  fallback to other numbers in the response. Undecoded byte-level token markers
  (`Ġ`, `Ċ`, `ĉ`) raise an error.
- `boxed_rate`: fraction of responses with a parseable final boxed number.

## Checkpoint checks

`--parent-tokenizer` names the dense parent. Its tokenizer is used for every
run, and before any weights are loaded `common/fairness.py` checks that the
checkpoint was compressed from it (token-embedding table equal to the parent's),
that the config declares `--expect-bits`, and that sampled weight sparsity is
within 0.02 of `--expect-sparsity`. A failing checkpoint is refused.
`--allow-unverified-parent` runs it anyway and marks the manifest
`fair_checkpoint: false`, which `check_pair.py` refuses.

`check_pair.py` rescores every response in both runs and compares documents,
prompts, generation arguments and protocol metadata. It fails when either run's
checkpoint checks failed or are missing, when the runs name different dense
parents, or when the baseline is not the dense parent. It does not check
compression calibration settings or truncation.

## Files written

Each `--output` directory gets `manifest.json` (protocol, model and parent
identity, checkpoint checks, status), `samples_gsm8k_checked.jsonl` (one line per
question with prompt, response and score) and `results.json` (lm-eval aggregate
output).

## Scorer tests

```bash
python -m unittest -v test_scoring.py
```
