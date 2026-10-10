# OmnizipEval_checking_results

The official evaluation codebases behind OmniZipEval (18 benchmarks), each copied at the version used, with clean scripts that state the dataset
version and every generation setting. Every script takes `MODEL` (a Hugging Face id or a local directory, dense
or compressed). It defaults to the dense original model, so scripts name only the original models.

| Codebase | Version | Benchmarks | Scripts | Settings and notes |
|---|---|---|---|---|
| [lm-evaluation-harness/](lm-evaluation-harness/OMNIZIPEVAL.md) | EleutherAI v0.4.12 (`6d64254`), unmodified | GSM8K, TruthfulQA, GPQA / GPQA Diamond | `gsm8k/evaluate.py`, `truthfulqa/evaluate.py`, `gpqa/evaluate.py` | task YAMLs and pins in each folder |
| [LiveCodeBench/](LiveCodeBench/scripts/omnizipeval/README.md) | upstream `28fef95` + local runner patches | LiveCodeBench `codegeneration`, `release_v4` (713 problems) | `scripts/omnizipeval/{qwen25_instruct,deepseek_r1_distill,qwen3}.sh`, `score.py` | per family: n, temperature, max_tokens, stop, context |
| [LongBench/](LongBench/scripts/omnizipeval/README.md) | THUDM `2e00731` + `\boxed{}` fork commit `1d4aa4c` | LongBench v2 (`THUDM/LongBench-v2@2b48e49`, 503 questions, CoT) | `scripts/omnizipeval/{qwen25_instruct_cot,deepseek_r1_distill_cot,qwen3_moe_cot}.sh` | per family: input cap, CoT/answer budget, sampling, thinking |
| [WorfBench/](WorfBench/scripts/omnizipeval/README.md) | zjunlp `38a6455` + vLLM runner | WorfBench, 8 tasks, node P/R/F1 | `scripts/omnizipeval/run_worfbench.sh` (1,024 tokens), `run_worfbench_long.sh` (32,768) | greedy, top-p 0.95, `optional_space` parser, pinned scorer |
| [VLMEvalKit/](VLMEvalKit/EVAL.md) | upstream `0916343` + local patches | POPE, MMBench, MMMU, ScreenSpot v2, Spatial457, RealWorldQA, MTVQA, MM-SafetyBench, MM-Math, We-Math | `scripts/eval/<dataset>.sh` (one per dataset, `FAMILY` = qwen25_vl / llava_next / llava_v15 / qwen3_vl_moe) | each script writes an explicit `--config` JSON: tokens, decoding, pixels, prompt mode, judge |
| [teval/](teval/README.md) | upstream T-Eval scorer (unmodified, `third_party/`) | T-Eval, 6 sub-capabilities | `teval/evaluate.py` | data pinned by `download_data.py` |
| [xstest/](xstest/README.md) | official `paul-rottger/xstest@d7bb5bd7` prompts | XSTest, harmful / harmless | `xstest/evaluate.py` | vLLM greedy, 1,024 tokens, Qwen3Guard judge |

`common/fairness.py` is used by the gsm8k, truthfulqa, gpqa, teval and xstest runners. It checks that a
compressed checkpoint was produced from the dense parent (identical token embeddings) and has the expected
bit width or sparsity.

Each codebase's README lists its upstream commit, every local change and why, and the settings table.
Run a dense model and its compressed variants with the same script and environment.

The repository contains source code, dataset settings and run instructions only.
