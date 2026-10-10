# lm-evaluation-harness in OmniZipEval

Unmodified copy of [EleutherAI/lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)
at tag **v0.4.12** (commit `6d642546f4688648fced259eb3302efd36ece5af`), without `.git`.

It is the harness behind three benchmark folders. Install it from the benchmark folder with
`pip install -r requirements.txt` (which installs `-e ../lm-evaluation-harness[hf|vllm]`).

| Benchmark | Folder | Task | Shots | Decoding / budget | Dataset version |
|---|---|---|---|---|---|
| GSM8K | `../gsm8k/` | `gsm8k_checked` (`gsm8k/task.yaml`, chat, demonstrations as turns) | 4 | greedy, `max_gen_toks` 4096, HF backend BF16 | `openai/gsm8k` `main` @ `740312a` |
| TruthfulQA | `../truthfulqa/` | `truthfulqa_mc1`, `truthfulqa_mc2` from `lm_eval/tasks/truthfulqa/` | 0 (task prompt holds 6 fixed QA pairs) | log-likelihood, no chat template | `truthfulqa/truthful_qa` `multiple_choice` @ `741b827` |
| GPQA, GPQA Diamond | `../gpqa/` | `gpqa_{main,diamond}_qwen25_report` (`gpqa/scripts/eval/tasks/gpqa_qwen25_report/`) | 0 | greedy; 8,192 tokens (Qwen2.5), 32,768 (R1-Distill, Qwen3); vLLM | `Idavidrein/gpqa` @ `83022ce` |
