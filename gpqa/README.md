# GPQA and GPQA Diamond

`evaluate.py` evaluates one checkpoint on GPQA Diamond and GPQA Main with lm-eval
and vLLM, using the tasks in `scripts/eval/tasks/gpqa_qwen25_report/`.
`check_pair.py` checks that a dense run and a compressed-variant run are
comparable. Protocol settings are constants in `evaluate.py`; only model
identity, output directory, expectations and hardware are options.

## Install

Python 3.10+ with a GPU build of vLLM matching the local CUDA/PyTorch. The
dataset `Idavidrein/gpqa` is gated: accept its terms on the Hub first. From this
folder:

```bash
python -m pip install -r requirements.txt   # lm-eval 0.4.12 from ../lm-evaluation-harness [vllm]
```

## Run

```bash
# 1. dense parent (--model and --parent are the same)
python evaluate.py --model Qwen/Qwen2.5-3B-Instruct --revision <sha> \
  --parent Qwen/Qwen2.5-3B-Instruct --parent-revision <sha> --output runs/qwen25_3b/dense

# 2. compressed variants, same --parent, with what each should be
python evaluate.py --model <org>/Qwen2.5-3B-Instruct-SparseGPT-50pct --revision <sha> \
  --parent Qwen/Qwen2.5-3B-Instruct --parent-revision <sha> --expect-sparsity 0.5 \
  --output runs/qwen25_3b/sparsegpt
python evaluate.py --model Qwen/Qwen2.5-3B-Instruct-GPTQ-Int4 --revision <sha> \
  --parent Qwen/Qwen2.5-3B-Instruct --parent-revision <sha> --expect-bits 4 \
  --output runs/qwen25_3b/gptq_int4
#   pruned 50%: --expect-sparsity 0.5   GPTQ/AWQ-Int4: --expect-bits 4   SmoothQuant W8A8: --expect-bits 8

# 3. pair check
python check_pair.py runs/qwen25_3b/dense runs/qwen25_3b/sparsegpt
```

Hub IDs need `--revision` / `--parent-revision` as full 40-character commit
SHAs; local directories need none. `--tensor-parallel-size` and
`--gpu-memory-utilization` are recorded in the manifest. `--smoke N` evaluates
the first N questions of each split and is recorded in the protocol, so
`check_pair.py` refuses to pair a smoke run with a full run. Existing output
directories are refused.

## Settings

| Setting | Value |
| --- | --- |
| Dataset | `Idavidrein/gpqa`, revision `83022cefff930aea54f654c0b282e74b9eeda5c6` |
| Splits | `gpqa_diamond` (198 questions) and `gpqa_main` (448), both run for every model |
| Tasks | `gpqa_diamond_qwen25_report`, `gpqa_main_qwen25_report` |
| Shots | 0 |
| Prompt | OpenAI simple-evals GPQA prompt (ends with `Answer: $LETTER`), no system prompt |
| Choice order | simple-evals permutation from `random.Random(0)`, one `rng.sample` per question |
| Chat template | the dense parent's tokenizer and chat template |
| Decoding | greedy (temperature 0, top-p 1, one sample) |
| Backend | lm-eval 0.4.12 + vLLM, bfloat16, no cache |
| Seeds | 0 (vLLM, random, numpy, torch, few-shot) |

The generation budget depends on the dense parent's family, which is read from
the parent checkpoint (config `model_type` and chat template), not from a flag.
Every variant uses its parent's budget:

| Parent family | max_gen_toks | max_model_len | Thinking |
| --- | ---: | ---: | --- |
| Qwen2.5 (Instruct) | 8,192 | 16,384 | n/a |
| DeepSeek-R1-Distill (Qwen, Llama) | 32,768 | 40,960 | on (template opens `<think>`) |
| Qwen3 | 32,768 | 40,960 | on (template default) |

## Metrics

Both filters read only the text after the last `</think>` when present.

- `exact_match,strict-match` (primary; `paper_metric` in the manifest): the
  first `Answer: X` match (simple-evals regex) must equal the correct letter.
- `exact_match,flexible-extract`: the last match of `Answer: X`, `\boxed{X}`,
  `The answer is X` or `(X)`, in that order of preference.

`evaluate.py` reports both as percentages per split, plus the number of
responses that reached `max_gen_toks` and the number with no strict-match answer.

## Checkpoint checks

Before loading a model, `common/fairness.py` checks that `--model` was
compressed from `--parent` (token-embedding table equal to the parent's), that
the config declares `--expect-bits`, and that sampled weight sparsity is within
0.02 of `--expect-sparsity`. A failing checkpoint is refused unless
`--allow-unverified-parent` is given, which marks the manifest
`fair_checkpoint: false`.

`check_pair.py` compares the protocol fingerprints and evaluated questions,
checks that the baseline is the dense parent and the variant names the same
parent, requires both checkpoints to have passed their checks, rescores every
response with both filters, and checks that both runs saw the same questions,
choice orders and rendered prompts.

## Files written

Each `--output` directory gets `manifest.json` (protocol, model and parent
identity, checkpoint checks, hardware, scores, status), `items.json` (evaluated
question IDs), `samples_gpqa_diamond_qwen25_report.jsonl` and
`samples_gpqa_main_qwen25_report.jsonl` (one line per question and filter) and
`results.json` (lm-eval aggregate output).
