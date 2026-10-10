# LongBench v2 for OmniZipEval

One launcher per model family for LongBench v2. Each launcher hard-codes the protocol, prints it and saves it to
`$OUTPUT/settings.txt`. It then starts a vLLM OpenAI-compatible server, runs `pred.py --cot` and scores the
predictions with `result.py`.

The code is [THUDM/LongBench](https://github.com/THUDM/LongBench) at `2e00731f8d0bff23dc4325161044d0ed8af94c1e`
with the changes listed under [Local changes](#local-changes-vs-upstream-2e00731). The image folders `misc/` and
`LongBench/misc/` are not included, so the image links of the top-level README do not resolve.

## Install

```bash
pip install vllm openai tiktoken datasets transformers torch tqdm   # a vLLM version that serves the model family
```

`requirements.txt` is upstream's pin list. The scripts only need a vLLM build with the OpenAI server, plus the
`pred.py` client dependencies above.

## Run

```bash
bash scripts/omnizipeval/qwen25_instruct_cot.sh          # dense Qwen/Qwen2.5-3B-Instruct
MODEL=Qwen/Qwen2.5-7B-Instruct bash scripts/omnizipeval/qwen25_instruct_cot.sh
bash scripts/omnizipeval/deepseek_r1_distill_cot.sh      # dense deepseek-ai/DeepSeek-R1-Distill-Llama-8B
bash scripts/omnizipeval/qwen3_moe_cot.sh                # dense Qwen/Qwen3-30B-A3B, GPUs 0,1,2,3

# a compressed checkpoint: the protocol is unchanged; the tokenizer and template come from the dense parent
MODEL=/ckpt/qwen25_7b_wanda50 PARENT=Qwen/Qwen2.5-7B-Instruct NAME=qwen25_7b_wanda50 GPUS=1 \
  bash scripts/omnizipeval/qwen25_instruct_cot.sh
```

Environment variables (none of them changes the protocol):

- `MODEL`: HF id or local path. The default is the dense model.
- `PARENT`: tokenizer and chat template for truncation and for the server. The default is `MODEL`. Set it to the dense
  model for compressed checkpoints.
- `NAME`: names the prediction file. The default is `basename MODEL`.
- `GPUS`, `TP` (default: number of GPUs), `PP` (default 1).
- `OUTPUT`: default `outputs/<script>/<NAME>`.
- `PYTHON`, `PORT`.
- `PRED_PASSES`: default 2. pred.py resumes missing ids on each pass.
- `DTYPE`: default `bfloat16`. Change it only for a kernel that cannot run BF16.

## Settings

All three scripts share these settings:

- Dataset: `THUDM/LongBench-v2` at revision `2b48e494f2c7a2f0af81aae178e05c7e1dde0fe9`, split `train`, all 503 questions.
- Two-stage CoT. Stage 1 sends `prompts/0shot_cot.txt` (document + question). Stage 2 sends
  `prompts/0shot_cot_ans.txt` (question + stage-1 output, with the document replaced by "omitted") and asks for
  `\boxed{X}`.
- Middle truncation, applied in both stages: if the rendered prompt is longer than `max_len` tokens
  (`tokenizer.encode`, `PARENT` tokenizer), keep the first and last `max_len/2` tokens and decode with
  `skip_special_tokens=True`.
- One user message, no system prompt. The server applies the `PARENT` chat template.
- A vLLM OpenAI server with `--generation-config vllm`, so checkpoint generation defaults never leak in.

| | `qwen25_instruct_cot.sh` | `deepseek_r1_distill_cot.sh` | `qwen3_moe_cot.sh` |
|---|---|---|---|
| Dense models | Qwen2.5-3B/7B-Instruct | DS-R1-Distill-Llama-8B, -Qwen-1.5B, -Qwen-7B | Qwen3-30B-A3B |
| Input cap (`max_len`) | 30,000 | 96,000 | 30,000 |
| `--max-model-len` | 32,768 | 131,072 | 40,960 |
| max new tokens (CoT / answer) | 1,024 / 128 | 2,048 / 2,048 (`-t 2048`) | 4,096 / 4,096 |
| temperature / top-p / top-k | 0.1 / 1.0 / 20 | 0.6 / 0.95 / off | 0.6 / 0.95 / 20 |
| repetition penalty | 1.05 | 1.0 | 1.0 |
| seed | none | none | 42, per question and stage |
| thinking | none | always (template opens `<think>`) | `enable_thinking=true`, both stages |
| dtype, GPU mem | bf16, 0.92 | bf16, 0.92 | bf16, 0.88, eager, `max_num_seqs` 4, `max_num_batched_tokens` 8192 |
| client processes | 8 | 8 | 4 |

For Qwen2.5, top-k 20 and repetition penalty 1.05 are the values of Qwen2.5-Instruct's `generation_config.json`. They
are sent explicitly with every request, so a compressed checkpoint's own `generation_config.json` cannot change them.
The seed of the Qwen3 script is sent per request as `sha256("42:<_id>:<cot|answer>") mod 2^31`, and the server also
gets `--seed 42`.

Budgets, input caps and sampling differ by family, and from the upstream LongBench v2 setup (120K input,
1,024 / 128 tokens, "The correct answer is" prompt), so scores are comparable only within a family.

## Metrics (`result.py`)

For each `*.jsonl` file in `--results_dir`:

- `acc = int(judge)`, where `judge = (pred == answer)`. An unparsed answer counts as wrong, because `compensated = False`.
- Overall = `100 * sum(acc) / number of records`.
- Easy = `difficulty == "easy"` and Hard = every other record (192 / 311 questions).
- Short / Medium / Long come from `length` (180 / 215 / 108 questions).
- Every value is `round(100 * correct / n, 1)`.

The denominator is the number of records in the file. pred.py drops a question whose request failed five times or
returned empty text, so the scripts resume until all 503 ids are present, and `result.py` warns otherwise.

`extract_answer` drops text up to the first `</think>`. It then tries "The correct answer is (X)",
"The correct answer is X", `\boxed{X}`, and finally the last standalone letter A-D.

## Files written in `$OUTPUT`

- `settings.txt`, `server_command.txt`
- `vllm_server.log`, `pred.log`: console output of the server and of pred.py
- `predictions/<NAME>_cot[_t2048].jsonl`: one record per question (`response_cot`, `response`, `pred`, `judge`)
- `result.txt`: Overall, Easy, Hard, Short, Medium, Long

## Local changes vs upstream 2e00731

- `prompts/0shot_cot_ans.txt` and `prompts/0shot.txt` ask for `\boxed{X}` instead of
  "The correct answer is (X)". `prompts/0shot_cot.txt`, `0shot_no_context.txt` and `0shot_rag.txt` are unchanged.
- `pred.py`:
  - `extract_answer` as described above.
  - The server URL and key come from `LONGBENCH_API_BASE` / `LONGBENCH_API_KEY`, and the client timeout is 600 s
    (`LONGBENCH_REQUEST_TIMEOUT`). Encoding and decoding go through tokenizer-agnostic helpers; tiktoken is used only
    for unregistered `gpt-*`/`o1*` names.
  - Requests send `top_p`. Model-name rules choose defaults: R1-Distill and Qwen3 models get temperature 0.6,
    top-p 0.95 and 4,096 tokens per stage; other models 0.1 / 1.0 / 1,024 + 128 with `--cot`, and a 1,024-token
    answer budget without it.
  - New options, each defaulting to the model-name rules: `--max_new_tokens/-t` (both stages; adds `_t{N}` to the
    output file name), `--model_path`, `--tokenizer`, `--served_model_name`, `--max_len` (any HF id or local
    checkpoint without editing `config/*.json`), `--cot_max_new_tokens`, `--ans_max_new_tokens`, `--temperature`,
    `--top_p`, `--top_k`, `--repetition_penalty`, `--enable_thinking` (sends `chat_template_kwargs`), `--seed N`
    (per-request seed, see above) and `--dataset_revision`.
  - pred.py prints the effective settings and the number of loaded questions.
- `result.py`: `--results_dir`, `--output` and `--expected`; reads one-line or blank-line JSONL files, prints `nan` for
  an empty group, warns on stderr when a file does not hold 503 questions and prints the table to stdout.
- `config/model2path.json` / `model2maxlen.json`: upstream entries plus the three DS-R1-Distill models (96,000-token
  input cap) and the official Qwen checkpoints `Qwen2.5-3B-Instruct` (+ `-AWQ`, `-GPTQ-Int4`) and `Qwen3-30B-A3B`
  (30,000). `Qwen2.5-7B-Instruct` is capped at 30,000 instead of 120,000 (Qwen2.5's native context is 32,768).
