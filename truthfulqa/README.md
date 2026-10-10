# TruthfulQA

`evaluate.py` evaluates one checkpoint on TruthfulQA MC1 and MC2 with the stock
lm-eval tasks (`lm_eval/tasks/truthfulqa/` in `../lm-evaluation-harness`).
`check_pair.py` checks that a dense run and a compressed-variant run are
comparable.

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
| Dataset | `truthfulqa/truthful_qa`, config `multiple_choice`, revision `741b8276f2d1982aa3d5b832d3ee81ed3b896490` |
| Split | `validation`, 817 questions |
| Tasks | `truthfulqa_mc1`, `truthfulqa_mc2` (lm-eval 0.4.12; task-file hashes recorded per run) |
| Shots | 0 additional; the task prompt itself contains 6 fixed QA pairs |
| Chat template | none |
| Scoring | log-likelihood of each candidate answer (no generation) |
| Context | 8,192 tokens |
| Backend | lm-eval HF backend, BF16 weights, float32 softmax, batch size 1 by default |
| Seeds | 0 (random, numpy, torch, few-shot); no response or request cache |

## Metrics

- MC1: 1 if the highest-likelihood candidate is the single true answer, else 0.
- MC2: normalized probability mass assigned to the true candidates.

Both are averaged over questions; `results.json` stores them on a 0–1 scale and
`check_pair.py` prints percentages.

## Checkpoint checks

`--parent-tokenizer` names the dense parent. Its tokenizer is used for every
run, and before any weights are loaded `common/fairness.py` checks that the
checkpoint was compressed from it (token-embedding table equal to the parent's),
that the config declares `--expect-bits`, and that sampled weight sparsity is
within 0.02 of `--expect-sparsity`. A failing checkpoint is refused.
`--allow-unverified-parent` runs it anyway and marks the manifest
`fair_checkpoint: false`, which `check_pair.py` refuses.

`check_pair.py` rescores every question in both runs from the stored candidate
log-likelihoods and compares documents, prompts and protocol metadata. It fails
when either run's checkpoint checks failed or are missing, when the runs name
different dense parents, or when the baseline is not the dense parent. It does
not check compression calibration settings.

## Files written

Each `--output` directory gets `manifest.json` (protocol, model and parent
identity, checkpoint checks, status), `samples_truthfulqa_mc1.jsonl` and
`samples_truthfulqa_mc2.jsonl` (one line per question with candidate
log-likelihoods and score) and `results.json` (lm-eval aggregate output).
