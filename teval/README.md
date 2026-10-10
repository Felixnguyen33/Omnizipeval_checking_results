# T-Eval

Tool-use evaluation with [T-Eval](https://github.com/open-compass/T-Eval) (English v2, all eight subsets).
It reports Overall and six sub-capabilities: Instruct, Plan, Reason, Retrieve, Understand and Review.
Each cohort is one dense model plus its compressed variants, and every model in it runs under the
identical protocol below. `check_pair.py` refuses a pair that does not.

```text
teval/
  evaluate.py        # runner: all protocol constants are in this file
  scoring.py         # subsets, answer extraction, upstream scorer calls, Overall
  check_pair.py      # pair fairness + full re-verification of both run folders
  download_data.py   # fetches the pinned data files and checks their sha256
  requirements.txt
  third_party/       # upstream T-Eval scorer (unmodified) + its Apache-2.0 LICENSE
  data/              # created by download_data.py (git-ignored)
```

## Install

Install a CUDA build of PyTorch that matches `requirements.txt`, then run:

```bash
pip install -r teval/requirements.txt
python teval/download_data.py        # 8 files, ~23 MB, from lovesnowbest/T-Eval at a pinned commit
```

No API key is needed. The scorer does not call an LLM judge.

## Run a cohort

Use the same environment for every model in a cohort. Package versions are recorded, and
`check_pair.py` fails a pair whose versions differ. Write outputs outside the repository.
Hub models need a 40-character commit (`--revision`, `--parent-revision`). A local snapshot
directory can be passed instead, without a revision.

```bash
P=Qwen/Qwen2.5-3B-Instruct; PREV=<40-char commit>
OUT=/path/outside/repo/teval

# dense parent (model == parent)
CUDA_VISIBLE_DEVICES=0 python teval/evaluate.py --model $P --revision $PREV \
    --parent $P --parent-revision $PREV --output $OUT/qwen2.5-3b-instruct

# pruned variant: the parent check and a sparsity check run before any GPU work
CUDA_VISIBLE_DEVICES=0 python teval/evaluate.py \
    --model <org>/<compressed-checkpoint> --revision <commit> \
    --parent $P --parent-revision $PREV --expect-sparsity 0.5 --output $OUT/qwen2.5-3b-sparsegpt

# quantized variant
CUDA_VISIBLE_DEVICES=0 python teval/evaluate.py --model Qwen/Qwen2.5-3B-Instruct-AWQ --revision <commit> \
    --parent $P --parent-revision $PREV --expect-bits 4 --output $OUT/qwen2.5-3b-awq

python teval/check_pair.py $OUT/qwen2.5-3b-instruct $OUT/qwen2.5-3b-sparsegpt
```

Before the model loads, `evaluate.py` compares the variant's token-embedding table with the parent's
(`common/fairness.py`). It refuses the run if the variant was compressed from a different checkpoint,
for example a base-model prune paired with an Instruct parent, or if the declared bits or measured
sparsity contradict `--expect-*`. It also refuses an existing `--output` folder.
`--tensor-parallel-size N` only spreads one model over N GPUs. It is recorded in the manifest, and
`check_pair.py` lists it under `hardware_differences`. Keep it the same within a cohort.

`--smoke N` evaluates the first N items of each subset to test the pipeline. It writes `"smoke": N`
into the protocol, so a smoke run never pairs with a full run. Use it for pipeline tests only.

## Protocol (constants in `evaluate.py` / `scoring.py`)

| Setting | Value |
| --- | --- |
| Data | `lovesnowbest/T-Eval` @ `af355ab2b62cdbae4262ac41c7529ffeae395012`, English v2. The sha256 of each file is pinned in `download_data.py` and recorded in the manifest |
| Subsets (prompt type) | instruct (json; the file mixes json and string items), review (str), plan (json), plan (str), reason (str), retrieve (str), understand (str), reason_retrieve_understand (json). These are the 8 runs of upstream `test_all_en.sh` |
| Items | full splits: 553 × 7 + 487 (review) = 4,358. Item id = `<subset>/<data key>` |
| Prompt | T-Eval `origin_prompt` messages, rendered with the **parent's** chat template (`apply_chat_template(add_generation_prompt=True)`) and tokenized by the parent tokenizer. The role `function` (tool output) is sent as `user`, as upstream T-Eval's Qwen and ChatGLM meta templates do. No lagent meta template is used |
| Decoding | greedy (temperature 0, top-p 1, no top-k, repetition penalty 1). Stop tokens are the parent tokenizer EOS plus the parent `generation_config.json` `eos_token_id`. Checkpoint generation configs are ignored (`generation_config="vllm"`) |
| Budget | 4,096 new tokens; context 16,384. The longest prompt is about 8.2k tokens, so no prompt is truncated (a run that would truncate is refused) |
| Backend | vLLM, bfloat16, seed 0, prefix caching off, `max_num_seqs` 256, `max_num_batched_tokens` 16,384, `gpu_memory_utilization` 0.85. One `generate` call over all items in fixed order. The quantization kernel is chosen by vLLM from the checkpoint config and recorded as `hardware.vllm_quantization` |
| Answer extraction | If the output contains `</think>`, keep only the text after the last one. If the parent template opened a `<think>` block and it was never closed, the prediction is empty (scored as a failure). Then upstream `split_special_tokens` is applied. For Qwen2.5-Instruct this only applies `split_special_tokens` |
| Scorer | upstream T-Eval evaluators, commit `58f22406404d7e2a4f36856a19c7f4dc28a0a5f0`, unmodified. Embeddings use `sentence-transformers/all-mpnet-base-v2` @ `e8c3b32edf5434bc2275fc9bab85f82640a19130` on CPU. Plan and Reason use embedding similarity; the other scores are exact match |

The manifest also records the parent's chat-template and vocabulary hashes, the hash of the full
prompt token set, source-file hashes and package versions.

### How Overall is computed

The seven numbers follow upstream `teval/utils/convert_results.py` (vendored; `check_pair.py` runs it
and compares). All values are ×100.

- Instruct = mean of (json_format + json_args_em)/2 and (string_format + string_args_em)/2, from `instruct_json`
- Plan = mean(plan_str.f1_score, plan_json.f1_score)
- Reason = mean(reason_str.thought, rru_json.thought)
- Retrieve = mean(retrieve_str.name, rru_json.name)
- Understand = mean(understand_str.args, rru_json.args)
- Review = review_str.review_quality (upstream averages this value with itself)
- **Overall = unweighted mean of the six**

Upstream `convert_results.py` prints them as `Cut Paste Results: [Overall, Instruct, Plan, Reason,
Retrieve, Understand, Review]`, the same order and formula.

## Outputs (in `--output`)

- `manifest.json`: protocol and its sha256, model/parent identity, checkpoint checks, hardware, generation stats, and `scores` (the seven numbers)
- `items.json`: the evaluated item ids
- `generations.jsonl`: one row per item, with the raw generation, finish reason, token counts, prompt hash and the extracted prediction
- `predictions/<subset>.json`: upstream `test.py` output format (data item + `prediction`), read by the evaluators
- `scores_by_subset.json`: upstream per-subset metrics, in the layout `convert_results.py` reads
- `run_info.json`: hardware and timing

## What `check_pair.py` verifies

1. `fairness.compare_manifests`: identical protocol (including package versions), same items, same
   dense parent, baseline is the parent, and both checkpoint checks passed.
2. Every item's prompt token ids are identical in the two runs.
3. For each run separately:
   - prompts are rebuilt from the pinned data with the parent tokenizer and must match the recorded hashes;
   - predictions are re-extracted from the raw generations and must match the saved ones;
   - `predictions/*.json` must equal the data item plus that prediction;
   - the upstream evaluators re-score them, and every per-subset metric and all seven numbers must
     equal the values in the run folder (tolerance 1e-6);
   - upstream `convert_results.py` must print the same numbers.

The exit code is 0 only if all of this holds.

## Third-party code

`third_party/teval/` is unmodified code from open-compass/T-Eval (Apache-2.0, see
`third_party/LICENSE`). `split_special_tokens` in `scoring.py` is copied verbatim from upstream
`test.py`. Upstream's lagent LLM wrappers are not used. vLLM is called directly.
