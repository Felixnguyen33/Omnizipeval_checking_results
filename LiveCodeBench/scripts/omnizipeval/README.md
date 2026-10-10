# LiveCodeBench for OmniZipEval

This folder holds one launcher per model family for the LiveCodeBench code
generation benchmark, run with the `lcb_runner` package in this copy (MIT).

- Upstream: <https://github.com/LiveCodeBench/LiveCodeBench>, commit
  `28fef95ea8c9f7a547c8329f2cd3d32b92c1fa24`, with the local changes listed below.
- Dataset: `livecodebench/code_generation_lite`, `release_v4`, Hub commit
  `0fe84c3912ea0c4d4a78037083943e8f0c4dd505`. There is **no date filter**:
  release_v4 has **713 problems**, contests dated 2023-05-07 to 2024-10-05
  (238 easy, 279 medium, 196 hard).

## Install

Python 3.11 with a CUDA build of PyTorch and vLLM. From the copy root:

```bash
pip install -e .        # pyproject.toml pins datasets<4 (needed for the loading script)
```

## Run

One script per configuration. Each script takes only the checkpoint and the
hardware from the environment; every protocol setting is a constant in the
script and is written to `<OUTPUT>/settings.txt` before generation.

```bash
# dense model (the default MODEL of each script)
GPUS=0,1 bash scripts/omnizipeval/qwen25_instruct.sh          # Qwen/Qwen2.5-7B-Instruct
GPUS=0,1 bash scripts/omnizipeval/deepseek_r1_distill.sh      # deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
GPUS=0,1 bash scripts/omnizipeval/qwen3.sh                    # Qwen/Qwen3-8B

# another dense model of the family
MODEL=Qwen/Qwen2.5-32B-Instruct GPUS=0,1,2,3 bash scripts/omnizipeval/qwen25_instruct.sh

# a compressed checkpoint (HF id or local directory) + its dense parent
MODEL=/path/to/Qwen2.5-7B-Instruct-wanda-50 PARENT=Qwen/Qwen2.5-7B-Instruct \
  GPUS=0 OUTPUT=runs/qwen25_7b/wanda bash scripts/omnizipeval/qwen25_instruct.sh
MODEL=Qwen/Qwen2.5-3B-Instruct-AWQ PARENT=Qwen/Qwen2.5-3B-Instruct DTYPE=float16 \
  GPUS=0 OUTPUT=runs/qwen25_3b/awq bash scripts/omnizipeval/qwen25_instruct.sh
```

| Variable | Meaning | Default |
| --- | --- | --- |
| `MODEL` | HF id or local directory of the checkpoint (dense or compressed) | the script's dense model |
| `PARENT` | dense original model; one of the script's parents. Selects the prompt style and the output name | `MODEL` if it is a parent, else required |
| `TOKENIZER` | tokenizer id or directory | `PARENT` |
| `GPUS` | devices; one vLLM engine uses all of them (TP x PP = #GPUs) | `0` |
| `TP_SIZE` | tensor-parallel size (must divide #GPUs) | 4, 2 or 1, whichever divides #GPUs first |
| `DTYPE` | `bfloat16` or `float16` | `bfloat16` |
| `OUTPUT` | run directory; refused if it already holds a finished run | `runs/<protocol>/<basename MODEL>` |
| `PYTHON` | interpreter with the dependencies | `python` |
| `GPU_MEMORY_UTILIZATION`, `NUM_PROCESS_EVALUATE` | hardware only | `0.90`, `12` |

Compressed variants (e.g. GBLM-Pruner, Wanda and SparseGPT at 50% unstructured
sparsity; AWQ 4-bit; GPTQ-Int4; SmoothQuant W8A8) run with `PARENT` set to
their dense model.

How a checkpoint that is not registered is run: `lcb_runner.runner.main`
looks `--model` up in `lcb_runner/lm_styles.py` (that entry fixes the prompt
style and the output folder name) and loads the weights from
`--local_model_path` when it is given (an HF id or a directory both work, vLLM
resolves either). The scripts therefore always call

```
python -m lcb_runner.runner.main --model $PARENT --local_model_path $MODEL --tokenizer $TOKENIZER ...
```

from inside `$OUTPUT`, so the compressed model gets the dense parent's prompt
and its files land in `$OUTPUT/output/<parent repr>/` without colliding with
other runs. Without the scripts, do the same by hand: pass the dense parent's
registered name as `--model` and the checkpoint as `--local_model_path`, and
start each checkpoint from its own working directory.

## Settings

All three scripts: scenario `codegeneration`, `release_v4`, no
`--start_date/--end_date` (713 problems), top-p 0.95, frequency/presence
penalty 0, `--evaluate` with a 6 s per-test timeout, vLLM `enforce_eager`,
no prefix caching, `--trust_remote_code`, generation cache off (`--use_cache`
not passed), tokenizer = dense parent, no seed.

| Script | Parents (dense models) | Prompt style | n | temperature | max_tokens | stop | max_model_len |
| --- | --- | --- | ---: | ---: | ---: | --- | ---: |
| `qwen25_instruct.sh` | Qwen2.5-3B/7B/32B-Instruct | `CodeQwenInstruct` | 10 | 0.2 | 2,000 | `###` | 16,384 |
| `deepseek_r1_distill.sh` | DeepSeek-R1-Distill-Qwen-1.5B, -Qwen-7B, -Llama-8B | `DeepSeekR1` | 1 | 0.2 | 32,768 | `<｜end▁of▁sentence｜>` | 40,960 |
| `qwen3.sh` | Qwen3-8B, Qwen3-30B-A3B (thinking) | `CodeQwenInstruct` | 1 | 0.6 | 32,768 | `<\|im_end\|>` | 40,960 |

`CodeQwenInstruct` is LiveCodeBench's hand-written ChatML prompt (system line
"You are a helpful assistant."), not the model's chat template. Qwen3 is not
given `/no_think`, so it reasons by default. `###` is the LiveCodeBench parser
default and is only used for the non-reasoning family; it would cut a
reasoning trace.

`DTYPE` is the one numerics choice left to the caller. Use `float16` for a
quantized checkpoint whose vLLM kernel does not run in bfloat16. It is written
to `settings.txt`.

## pass@1

pass@1 = mean over the 713 problems of (correct samples / n). With n=1 it is
the fraction of problems solved; with n=10 it is the mean success rate of the
ten samples (not "any of ten"). A sample is correct only if it passes all
public and private tests. After generation the scripts run

```bash
python scripts/omnizipeval/score.py <OUTPUT>/output/<repr>/Scenario.codegeneration_<n>_<T>_eval_all.json \
  --expected-problems 713 --n <n> --out <OUTPUT>/score.json
```

which prints overall / easy / medium / hard pass@1 (in %), the contest date
range and the number of empty extracted programs, and fails if the run does
not have exactly 713 problems with n graded samples each. The same overall
number is the `Pass@1:` line of
`python -m lcb_runner.evaluation.compute_scores --eval_all_file <file>`
(ignore that tool's `Pass@k` lines for k > n: the estimator returns 1.0 there).

## Files written in `$OUTPUT`

- `settings.txt`: the protocol of the run
- `run.log`: console output of `lcb_runner`
- `output/<parent repr>/Scenario.codegeneration_<n>_<T>*.json`: generations,
  extracted code and per-test grading written by `lcb_runner`
- `score.json`: the output of `score.py`

## Local modifications (vs upstream 28fef95)

- `lcb_runner/runner/parser.py`: new flags `--tokenizer`,
  `--pipeline_parallel_size`, `--max_model_len`, `--gpu_memory_utilization`
  (help text of `--stop` extended). They evaluate a checkpoint with the parent
  tokenizer, fit 32B / 30B-A3B models on 24 GB GPUs, and give reasoning models
  a 40,960-token context.
- `lcb_runner/runner/vllm_runner.py`: passes those flags to vLLM; drops empty
  stop strings; if a sample contains at least three raw byte-level BPE markers
  (`Ġ`, `Ċ`) it is re-decoded from its token ids with the HF tokenizer
  (fallback: marker replacement).
- `lcb_runner/utils/extraction_utils.py`: code extraction for reasoning
  outputs. For `DeepSeekR1` style, or any output containing `<think>` /
  `</think>`, the text after the last `</think>` is used (an `<answer>` block
  first, then the last fenced block, preferring a `python` fence), falling back
  to fences anywhere in the output. Non-reasoning outputs use the last fenced
  block, as upstream. Byte-level markers are repaired before extraction.
- `lcb_runner/benchmarks/code_generation.py`: optional environment variable
  `LCB_DATASET_REVISION` passed as `revision=` to `load_dataset` (unset means
  upstream behaviour; the scripts set it to `0fe84c3`), and progress printing
  while the test cases are decoded.
- `lcb_runner/runner/main.py`: one start-up log line.
- `lcb_runner/lm_styles.py`: registrations of `Qwen/Qwen2.5-3B-Instruct`,
  `Qwen/Qwen3-8B`, `Qwen/Qwen3-30B-A3B` and the official Qwen AWQ / GPTQ-Int4
  checkpoints (`Qwen/Qwen2.5-{3B,32B}-Instruct-{AWQ,GPTQ-Int4}`,
  `Qwen/Qwen3-8B-AWQ`), all `CodeQwenInstruct`. Compressed checkpoints are run
  through `--local_model_path` instead (above).
- `pyproject.toml`: `datasets>=3.2.0,<4.0.0` (datasets 4 cannot run the
  dataset's loading script).
- `scripts/omnizipeval/` (this folder) and `runs/` in `.gitignore`.
