# Running the OmniZipEval VLM datasets with VLMEvalKit

This folder is [VLMEvalKit](https://github.com/open-compass/VLMEvalKit) at upstream commit
`09163430240669be7171e419bdf2f390f938d059` with the local changes listed at the end. Credentials (`.env`) and
local checkpoint paths are not included. Upstream's README and license are kept.

## Install and run

```bash
pip install -r requirements.txt && pip install -e .     # plus vllm for the vLLM rows below
FAMILY=qwen25_vl GPUS=0,1 bash scripts/eval/pope.sh                              # dense Qwen2.5-VL-3B
FAMILY=qwen25_vl MODEL=<HF id or local dir> NAME=qwen25vl3b_wanda50 GPUS=0,1 bash scripts/eval/pope.sh
FAMILY=qwen25_vl MODEL=Qwen/Qwen2.5-VL-32B-Instruct GPUS=0,1,2,3 NPROC=1 bash scripts/eval/mmbench.sh
FAMILY=qwen3_vl_moe GPUS=0,1,2,3,4,5,6 bash scripts/eval/realworldqa.sh
DRY_RUN=1 FAMILY=llava_v15 bash scripts/eval/mmmu.sh       # only write and print the config
```

There is one script per dataset in `scripts/eval/`. Each script builds a VLMEvalKit `--config` JSON in which
every model argument is written out and runs `run.py --config`. Nothing comes from the registrations in
`vlmeval/config.py`.

| Variable | Meaning | Default |
|---|---|---|
| `FAMILY` | `qwen25_vl`, `llava_next`, `llava_v15` or `qwen3_vl_moe`; selects the model class and every generation setting | `qwen25_vl` |
| `MODEL` | Hugging Face id or local directory, dense or compressed. Only the weights change with it | dense model of the family (below) |
| `PROCESSOR` | LLaVA families only: processor directory or id (`processor_path`) | the dense LLaVA model |
| `NAME` | model key in the config, used for output file names | basename of `MODEL` |
| `GPUS`, `NPROC` | visible GPUs; transformers/Qwen2.5-VL-vLLM rows run `NPROC` processes and split `GPUS` evenly between them | `0`; `NPROC` = number of GPUs |
| `WORK_DIR`, `MASTER_PORT`, `PYTHON` | output directory, torchrun port, interpreter | `outputs/<dataset>`, 29500, `python` |
| `JUDGE_BASE_URL`, `JUDGE_MODEL`, `JUDGE_KEY` | MM-SafetyBench only: local OpenAI-compatible judge | unset, `Qwen/Qwen3-30B-A3B`, `EMPTY` |
| `DRY_RUN` | `1`: write and print the config, do not run | unset |

Extra arguments go to `run.py` (for example `--mode infer` or `--reuse`). No API judge is used: every script
passes `--judge exact_matching`, except MM-SafetyBench, which needs a local judge server.

### Families

| `FAMILY` | Class | Dense model (pinned revision) | Backend |
|---|---|---|---|
| `qwen25_vl` | `Qwen2VLChat` | `Qwen/Qwen2.5-VL-3B-Instruct` (`66285546d2b821cf421d4f5eb2576359d3770cd3`); 7B `cc594898137f460bfe9f0759e9844b3ce807cfb5`; 32B `7cfb30d71a1f4f49a57592323337a4a4727301da` | transformers BF16; vLLM for MM-Math and WeMath |
| `llava_next` | `LLaVA_Next` | `llava-hf/llava-v1.6-vicuna-7b-hf` (`c916e6cdcd760b4cecd1dd4907f84ac649f93b23`) | transformers FP16 |
| `llava_v15` | `LLaVA_Next` (loads `model_type=llava` checkpoints with the LLaVA-1.5 classes, see below) | `llava-hf/llava-1.5-7b-hf` (`b234b804b114d9e37bb655e11cbbb5f5e971b7a9`) | transformers FP16 |
| `qwen3_vl_moe` | `Qwen3VLChat` (`use_vllm`) | `Qwen/Qwen3-VL-30B-A3B-Instruct` (`9c4b90e1e4ba969fd3b5378b57d966d725f1b86c`) | vLLM BF16, TP 1, pipeline parallel over `GPUS` |

A Hub id resolves to the latest revision. To pin a revision, pass the local snapshot directory as `MODEL`.

## Settings per dataset and family

Shorthand used in the table:

- **Q-pix**: Qwen2.5-VL image bounds `min_pixels=1003520`, `max_pixels=12845056` (1280..16384 x 28 x 28).
- **Q3-pix**: Qwen3-VL bounds `min_pixels=4096`, `max_pixels=16777216` (4..16384 x 32 x 32), also passed to vLLM's processor.
- **top-1**: Qwen2VLChat's default decoding: `do_sample=True, top_k=1, top_p=0.001, temperature=0.01,
  repetition_penalty=1.0`. With `top_k=1` this is greedy except for exact BF16 logit ties.
- **greedy**: `do_sample=False` (LLaVA: `num_beams=1`); in vLLM `temperature=0` (Qwen3: `top_p=1, top_k=-1,
  presence_penalty=0`), seed 0.
- **Q3-vLLM**: `dtype=bfloat16, tensor_parallel_size=1, pipeline_parallel_size=#GPUS, max_model_len=32768,
  enforce_eager, enable_expert_parallel=False, gpu_memory_utilization=0.85, max_num_batched_tokens=8192`,
  `hf_overrides` text architecture `Qwen3MoeForCausalLM` (needed by vLLM 0.27), one image per prompt except MMMU.
- **Q-vLLM**: `Qwen2VLChat(use_vllm=True)`: `dtype=bfloat16, seed=0, max_model_len=32768, enforce_eager,
  gpu_memory_utilization=0.85`, one image per prompt, one engine per process with TP 1.
- LLaVA rows use the processor of the dense model; images follow the LLaVA processor (no pixel arguments).
- Prompt: `dataset` = the dataset's own `build_prompt` (`use_custom_prompt=false`); `Qwen MCQ` = Qwen2VLChat's
  MCQ prompt (`use_custom_prompt=true`); `LLaVA MCQ` = LLaVA_Next's MCQ prompt (always on for MCQ datasets);
  `Qwen3` = Qwen3-VL prompt mixin (`use_custom_prompt=true`). No system prompt anywhere.

| Dataset (VLMEvalKit name, size) | Family | max_new_tokens | Decoding | Pixels | Prompt | Judge | Backend |
|---|---|---|---|---|---|---|---|
| POPE (`POPE`, 5,127 prompts) | qwen25_vl | 2048 | top-1 | Q-pix | dataset | exact_matching | transformers |
| POPE | llava_next, llava_v15 | 2048 | greedy | processor | dataset | exact_matching | transformers |
| POPE | qwen3_vl_moe | 128 | greedy | Q3-pix | Qwen3 | exact_matching | Q3-vLLM |
| MMBench (`MMBench`, v1.0 EN, `test` row: 1,784 questions) | qwen25_vl | 2048 | top-1 | Q-pix | dataset | exact_matching | transformers |
| MMBench (`MMBench`, `test` row) | llava_next, llava_v15 | 2048 | greedy | processor | LLaVA MCQ | exact_matching | transformers |
| MMBench (`MMBench_DEV_EN_V11`, v1.1 EN dev: 1,292 questions) | qwen3_vl_moe | 128 | greedy | Q3-pix | Qwen3 | exact_matching | Q3-vLLM |
| MMMU (`MMMU_DEV_VAL`, `validation` row: 900) | qwen25_vl | 2048 | top-1 | Q-pix | dataset | exact_matching | transformers |
| MMMU (`MMMU_DEV_VAL`) | llava_next, llava_v15 | 2048 | greedy | processor | LLaVA MCQ | exact_matching | transformers |
| MMMU (`MMMU_DEV_VAL`) | qwen3_vl_moe | 4096 | greedy | Q3-pix | Qwen3 (MMMU prompt, all images) | exact_matching | Q3-vLLM (image limit = class default) |
| ScreenSpot v2 (`ScreenSpot_v2` = Desktop/Mobile/Web, 1,272) | qwen25_vl | 2048 | top-1 | Q-pix | dataset (pyautogui), `qwen25_screenspot_adapter=false` | exact_matching (point-in-box) | transformers |
| ScreenSpot v2 | llava_next, llava_v15 | 2048 | greedy | processor | dataset | exact_matching | transformers |
| ScreenSpot v2 | qwen3_vl_moe | 128 | greedy | Q3-pix | Qwen3 `point_2d` (0-1000) prompt | built-in scorer after conversion | `scripts/run_qwen3_vl_suite.py` (vLLM, PP, batch 16) |
| Spatial457 (`Spatial457`, 23,752) | qwen25_vl | 2048 | top-1 | Q-pix | dataset | built-in | transformers |
| Spatial457 | llava_next, llava_v15 | 2048 | greedy | processor | dataset | built-in | transformers |
| RealWorldQA (`RealWorldQA`, 765) | qwen25_vl | 2048 | top-1 | Q-pix | dataset | exact_matching | transformers |
| RealWorldQA | llava_next, llava_v15 | 2048 | greedy | processor | LLaVA MCQ | exact_matching | transformers |
| RealWorldQA | qwen3_vl_moe | 128 | greedy | Q3-pix | Qwen3 | exact_matching | Q3-vLLM |
| MTVQA (`MTVQA_TEST`, 9 languages, 6,778) | qwen25_vl | 2048 | top-1 | Q-pix | dataset | built-in | transformers |
| MTVQA | llava_next, llava_v15 | 2048 | greedy | processor | dataset | built-in | transformers |
| MM-SafetyBench (`MMSafetyBench`, 1,680) | qwen25_vl | 2048 | top-1 | Q-pix | dataset | local LLM judge | transformers |
| MM-SafetyBench | qwen3_vl_moe | 2048 | greedy | Q3-pix | dataset (`use_custom_prompt=false`) | local LLM judge | Q3-vLLM |
| MM-Math (`MM-Math`, 5,901) | qwen25_vl | 16000 | greedy (vLLM T=0) | Q-pix | dataset | symbolic scorer | Q-vLLM |
| MM-Math | llava_next, llava_v15 | 16000 | greedy | processor | dataset | symbolic scorer | transformers |
| WeMath (`WeMath`, 1,740) | qwen25_vl | 16000 | greedy (vLLM T=0) | Q-pix | Qwen MCQ | exact_matching | Q-vLLM |
| WeMath | llava_next, llava_v15 | 16000 | greedy | processor | LLaVA MCQ | exact_matching | transformers |

MMBench uses v1.0 (`test` row) for Qwen2.5-VL and LLaVA and v1.1 dev for Qwen3-VL; the two versions are not
comparable. MMBench is scored circularly (a question counts only if all its option rotations are correct).

### MM-SafetyBench judge

Fixed in the script: `--judge $JUDGE_MODEL --judge-base-url $JUDGE_BASE_URL
--judge-args '{"temperature": 0, "max_tokens": 128, "seed": 0, "chat_template_kwargs": {"enable_thinking": false}}'`.
Judge model: `Qwen/Qwen3-30B-A3B` revision `ad44e777bcd18fa416d9da3bd8f70d33ebb85d39`, BF16, `max_model_len` 8192,
thinking disabled, greedy, 128 tokens, single user turn, VLMEvalKit's judge prompts. Serve it with e.g.
`vllm serve Qwen/Qwen3-30B-A3B --revision ad44e777... --dtype bfloat16 --max-model-len 8192 --seed 0 --port 8000`
and set `JUDGE_BASE_URL=http://127.0.0.1:8000/v1`.

### Metrics

Each script echoes what to read as `report=`:

- POPE: F1 per split (adversarial / popular / random) and `Overall` in `*_POPE_score.csv` (the `acc` column is
  accuracy).
- MMBench: `Overall` of the `test` (v1.0) or `dev` (V11) row of `*_acc.csv`.
- MMMU: `Overall` of the `validation` row (both splits are inferred).
- ScreenSpot v2: mean of the Desktop / Mobile / Web `Overall_Accuracy` (point-in-box).
- Spatial457: `score` (x100), per level `L*_score`. RealWorldQA: `Overall`. MTVQA: `Average`.
- MM-SafetyBench: `safety_rate` (x100) of the `Overall` row (100 - attack success rate). Check that `unknown`
  is 0: VLMEvalKit counts unknown labels as safe.
- MM-Math: `overall` (x100), symbolic answer comparison.
- WeMath: VLMEvalKit writes `Score (Strict/Loose)` and the four-dimension rates. The OmniZipEval WeMath metric
  is letter accuracy over the 1,740 questions: an answer letter is extracted from each prediction (rule
  `tail4_loose_stated_then_can_infer_else_full_strict`) and compared with the gold letter. It is computed
  outside VLMEvalKit from the prediction file the script writes.

### Family notes

- **LLaVA-v1.5-7B.** `FAMILY=llava_v15` runs `llava-hf/llava-1.5-7b-hf` (and llava-hf exports of its compressed
  variants) through `LLaVA_Next`, not `liuhaotian/llava-v1.5-7b` through `LLaVA`. Unmodified `LLaVA_Next` cannot
  run that checkpoint under transformers 5 (`LlavaNextProcessor` asks for `image_sizes`, which LLaVA-1.5's
  `CLIPImageProcessor` does not return). The local change loads `model_type=llava` checkpoints with
  `LlavaForConditionalGeneration` + `AutoProcessor` (CLIP 336 px, 576 image tokens, the checkpoint's chat
  template); `LLaVA_Next`'s prompt, generation and output-parsing code is unchanged.
- **Qwen2.5-VL MM-Math and WeMath.** These use `Qwen2VLChat(use_vllm=True)`: BF16, greedy (temperature 0),
  16,000 new tokens, 32,768-token context, seed 0, eager mode, Q-pix (applied by `qwen_vl_utils` and by vLLM's
  processor), one engine per GPU (TP 1). Qwen2VLChat sends one request at a time and keeps its own
  `max_num_seqs=5`. Setting `"use_vllm": false` in the generated config runs the transformers backend with the
  same decoding arguments.
- **Qwen2.5-VL ScreenSpot.** The script sets `qwen25_screenspot_adapter=false`, so VLMEvalKit's native
  pyautogui prompt and parser are used instead of the local `computer_use` adapter.
- **Qwen3-VL ScreenSpot.** `Qwen3VLChat` has no `point_2d` prompt, so `screenspot_v2.sh` runs
  `scripts/run_qwen3_vl_suite.py infer` instead of `run.py`: prompt "Locate the UI element described as: ...
  Return the center of the element as JSON: {"point_2d": [x, y]}", coordinates / 1000 converted to fractions
  for the same point-in-box scorer; an unparsable answer is a miss.
- **Qwen3-VL through `Qwen3VLChat` vs the suite.** Same engine arguments; `Qwen3VLChat` sends one request at a
  time (the suite batches 16 and uses prefix caching). SmoothQuant W8A8 Qwen3-VL checkpoints need the dynamic
  INT8 MoE fix: add `"worker_extension_cls": "qwen3_vl_int8_runtime.DynamicInt8MoEWorker"` to `vllm_kwargs` and
  put `scripts/` on `PYTHONPATH` (the suite and MMMU runners take `--dynamic-int8-moe-fix`).

## Files written by a run

- `$WORK_DIR/<NAME>_<DATASET>_config.json`: the generated `run.py --config` (for Qwen3-VL ScreenSpot, a record
  of the suite settings).
- `$WORK_DIR/<NAME>_<DATASET>_settings.txt`: the echoed settings, command and config.
- VLMEvalKit's predictions and score files under `$WORK_DIR/<NAME>/` (file names as in `report=`).
- Qwen3-VL ScreenSpot: `$WORK_DIR/<NAME>/` with the suite's checkpoints and `ScreenSpot_v2.metrics.json`
  (Desktop / Mobile / Web `Overall_Accuracy` and pooled accuracy).

## Local changes to upstream

- `run.py`: `LOCAL_RANK` defaults to 0, and the GPU split no longer asserts when a process already sees fewer
  GPUs than `LOCAL_WORLD_SIZE` (vLLM child processes).
- `vlmeval/dataset/mmmath.py`: `track_progress_rich(..., use_process=True)`. The symbolic judge uses
  signal-based timeouts, which do not work in threads, so it runs in processes.
- `vlmeval/dataset/spatial457.py`: `from tqdm import tqdm` (upstream's `import tqdm` made `evaluate()` fail).
- `vlmeval/dataset/utils/wemath.py`: `pd.concat` instead of `DataFrame._append`, which pandas 3 removed
  (same rows appended, formulas unchanged).
- `vlmeval/vlm/qwen2_vl/` (`model.py`, `prompt.py`): SDPA fallback when `flash_attn` is missing; a Qwen2.5-VL
  ScreenSpot `computer_use` adapter (on by default, switched off by `qwen25_screenspot_adapter=false`);
  optional `vllm_kwargs` merged into the vLLM engine arguments.
- `vlmeval/vlm/qwen3_vl/model.py`: vLLM tensor-parallel size chosen to divide the attention heads (or
  `tensor_parallel_size`); optional `vllm_kwargs` merged into the vLLM engine arguments.
- `vlmeval/vlm/llava/llava.py`: optional `processor_path`; sampling arguments dropped when `do_sample=False`;
  keyword arguments for the processor call (transformers 5); `model_type=llava` (LLaVA-1.5) checkpoints
  loaded with `LlavaForConditionalGeneration` + `AutoProcessor`.
- New `scripts/run_qwen3_vl_suite.py` (RealWorldQA, MMBench_DEV_EN_V11, ScreenSpot_v2, MMSafetyBench) and
  `scripts/run_qwen3_vl_mmmu.py`: resumable, batched Qwen3-VL runners on vLLM with VLMEvalKit datasets and
  scorers; `scripts/qwen3_vl_int8_runtime.py` / `qwen3_vl_mmmu_int8.py`: the dynamic INT8 MoE worker extension
  for vLLM 0.27.1. `screenspot_v2.sh` calls the suite.

## Known limits

- ScreenSpot v2 normalizes coordinates by the original screenshot size; a response without coordinates parses
  as (0, 0). This applies equally to dense and compressed models.
- MM-SafetyBench is VLMEvalKit's packaging (1,680 items, one SD image + the original question each), not the
  original release's 5,040 image-question pairs (SD, SD+TYPO, TYPO).
