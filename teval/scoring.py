"""T-Eval subsets, answer extraction and scoring, shared by ``evaluate.py`` and ``check_pair.py``.

Scoring uses the unmodified upstream T-Eval evaluators in ``third_party/teval``
(open-compass/T-Eval commit 58f22406404d7e2a4f36856a19c7f4dc28a0a5f0, Apache-2.0).
"""
from __future__ import annotations

import contextlib
import functools
import hashlib
import io
import json
import math
import re
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "third_party"))
# Upstream reason_retrieve_understand_evaluator.py does `import evaluate` (Hugging Face `evaluate`) but never
# uses it. Our runner is also called evaluate.py, so register an empty placeholder instead of importing either.
sys.modules.setdefault("evaluate", types.ModuleType("evaluate"))

from download_data import DATA_DIR, DATA_REPO, DATA_REVISION, DATA_SHA256  # noqa: E402

UPSTREAM_TEVAL = {"repo": "https://github.com/open-compass/T-Eval",
                  "commit": "58f22406404d7e2a4f36856a19c7f4dc28a0a5f0"}

# The eight English v2 runs of upstream test_all_en.sh, in its order. ``key`` is the name upstream test.py
# stores the result under (``<eval>_<prompt_type>``), which is what convert_results.py reads.
SUBSETS = (
    {"key": "instruct_json", "file": "instruct_v2.json", "eval": "instruct", "prompt_type": "json",
     "evaluator": "InstructEvaluator"},
    {"key": "review_str", "file": "review_str_v2.json", "eval": "review", "prompt_type": "str",
     "evaluator": "ReviewEvaluator"},
    {"key": "plan_json", "file": "plan_json_v2.json", "eval": "plan", "prompt_type": "json",
     "evaluator": "PlanningEvaluator"},
    {"key": "plan_str", "file": "plan_str_v2.json", "eval": "plan", "prompt_type": "str",
     "evaluator": "PlanningEvaluator"},
    {"key": "reason_str", "file": "reason_str_v2.json", "eval": "reason", "prompt_type": "str",
     "evaluator": "ReasonRetrieveUnderstandEvaluator"},
    {"key": "retrieve_str", "file": "retrieve_str_v2.json", "eval": "retrieve", "prompt_type": "str",
     "evaluator": "ReasonRetrieveUnderstandEvaluator"},
    {"key": "understand_str", "file": "understand_str_v2.json", "eval": "understand", "prompt_type": "str",
     "evaluator": "ReasonRetrieveUnderstandEvaluator"},
    {"key": "rru_json", "file": "reason_retrieve_understand_json_v2.json", "eval": "rru", "prompt_type": "json",
     "evaluator": "ReasonRetrieveUnderstandEvaluator"},
)
FULL_SPLIT_SIZES = {"instruct_json": 553, "review_str": 487, "plan_json": 553, "plan_str": 553, "reason_str": 553,
                    "retrieve_str": 553, "understand_str": 553, "rru_json": 553}

# Sentence embedding used by the Plan and Reason scorers. Upstream passes the unpinned name
# "all-mpnet-base-v2"; we pin the Hub commit and run it on CPU so scoring does not depend on the GPU.
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
EMBEDDING_REVISION = "e8c3b32edf5434bc2275fc9bab85f82640a19130"
EMBEDDING_DEVICE = "cpu"

CAPABILITIES = ("Overall", "Instruct", "Plan", "Reason", "Retrieve", "Understand", "Review")
OVERALL_RULE = (
    "Upstream T-Eval teval/utils/convert_results.py: "
    "Instruct = mean((instruct_json.json_format_metric + instruct_json.json_args_em_metric)/2, "
    "(instruct_json.string_format_metric + instruct_json.string_args_em_metric)/2); "
    "Plan = mean(plan_str.f1_score, plan_json.f1_score); Reason = mean(reason_str.thought, rru_json.thought); "
    "Retrieve = mean(retrieve_str.name, rru_json.name); Understand = mean(understand_str.args, rru_json.args); "
    "Review = review_str.review_quality; Overall = unweighted mean of the six; all x100."
)

THINK_END = "</think>"
POSTPROCESS_RULE = (
    "If the generation contains '</think>', keep only the text after the last '</think>'. Otherwise, if the "
    "parent chat template opened a reasoning block (rendered prompt ends with '<think>'), the reasoning was "
    "not finished and the prediction is empty. Then apply upstream test.py split_special_tokens."
)


# ----------------------------------------------------------------------------- data

def load_items(smoke: int | None = None) -> list[dict]:
    """All evaluated items in a fixed order: subsets in SUBSETS order, items in data-file order."""
    items = []
    for subset in SUBSETS:
        data = json.loads((DATA_DIR / subset["file"]).read_text())
        if not smoke and len(data) != FULL_SPLIT_SIZES[subset["key"]]:
            raise SystemExit(f"{subset['file']}: expected {FULL_SPLIT_SIZES[subset['key']]} items, found {len(data)}")
        keys = list(data)[:smoke] if smoke else list(data)
        items += [{"id": f"{subset['key']}/{k}", "subset": subset["key"], "key": k, "datum": data[k]} for k in keys]
    return items


def messages(datum: dict) -> list[dict]:
    """T-Eval prompt as chat messages. The 'function' role (tool output) is sent as a user turn, as upstream
    T-Eval's Qwen and ChatGLM meta templates do; Hugging Face chat templates have no 'function' role."""
    return [{"role": "user" if m["role"] == "function" else m["role"], "content": m["content"]}
            for m in datum["origin_prompt"]]


def token_ids_sha256(ids: list[int]) -> str:
    return hashlib.sha256(json.dumps(list(ids)).encode()).hexdigest()


# ----------------------------------------------------------------------------- answer extraction

def split_special_tokens(text):
    """Verbatim from upstream T-Eval test.py (commit 58f22406)."""
    text = text.split('<eoa>')[0]
    text = text.split('<TOKENS_UNUSED_1>')[0]
    text = text.split('<|im_end|>')[0]
    text = text.split('\nuser')[0]
    text = text.split('\nassistant')[0]
    text = text.split('\nUSER')[0]
    text = text.split('[INST]')[0]
    text = text.split('<|user|>')[0]
    text = text.strip()
    if text.startswith('```json'):
        text = text[len('```json'):]
    text = text.strip('`').strip()
    return text


def extract_prediction(raw: str, template_opened_think: bool) -> tuple[str, bool | None]:
    """Return (prediction, reasoning_closed); see POSTPROCESS_RULE."""
    if THINK_END in raw:
        return split_special_tokens(raw.rsplit(THINK_END, 1)[1]), True
    if template_opened_think:
        return "", False
    return split_special_tokens(raw), None


def template_opens_think(prompt_text: str) -> bool:
    return re.search(r"<think>\s*$", prompt_text) is not None


# ----------------------------------------------------------------------------- scoring

@functools.lru_cache(maxsize=1)
def embedding_model_path() -> str:
    from huggingface_hub import snapshot_download
    return snapshot_download(EMBEDDING_MODEL, revision=EMBEDDING_REVISION)


def _to_float(value):
    return float(value) if value is not None else None


def score_subset(subset: dict, predictions_file: Path) -> dict:
    """Run the upstream evaluator exactly as upstream test.py constructs it."""
    import numpy as np
    import random
    import torch
    from sentence_transformers import SentenceTransformer
    import teval.evaluators as evaluator_factory
    from teval.evaluators import planning_evaluator, reason_retrieve_understand_evaluator

    random.seed(0), np.random.seed(0), torch.manual_seed(0)
    cpu_model = functools.partial(SentenceTransformer, device=EMBEDDING_DEVICE)
    planning_evaluator.SentenceTransformer = cpu_model
    reason_retrieve_understand_evaluator.SentenceTransformer = cpu_model
    with contextlib.redirect_stdout(io.StringIO()):  # upstream prints the model path and item counts
        evaluator = getattr(evaluator_factory, subset["evaluator"])(
            str(predictions_file), default_prompt_type=subset["prompt_type"], eval_type=subset["eval"],
            bert_score_model=embedding_model_path())
        result = evaluator.evaluate()
    return {k: _to_float(v) for k, v in dict(result).items()}


def score_folder(predictions_dir: Path) -> dict:
    """{subset key: upstream metric dict} for every predictions/<key>.json in a run folder."""
    return {s["key"]: score_subset(s, predictions_dir / f"{s['key']}.json") for s in SUBSETS}


def capabilities(results: dict) -> dict:
    """The seven reported numbers (percent), computed as upstream convert_results.py does."""
    import numpy as np
    ins = results["instruct_json"]
    instruct = [(ins["json_format_metric"] + ins["json_args_em_metric"]) / 2,
                (ins["string_format_metric"] + ins["string_args_em_metric"]) / 2]
    plan = [results["plan_str"]["f1_score"], results["plan_json"]["f1_score"]]
    reason = [results["reason_str"]["thought"], results["rru_json"]["thought"]]
    retrieve = [results["retrieve_str"]["name"], results["rru_json"]["name"]]
    understand = [results["understand_str"]["args"], results["rru_json"]["args"]]
    review = [results["review_str"]["review_quality"], results["review_str"]["review_quality"]]
    final = [float(np.mean(x)) for x in (instruct, plan, reason, retrieve, understand, review)]
    final.insert(0, float(np.mean(final)))
    return {name: value * 100 for name, value in zip(CAPABILITIES, final)}


def upstream_convert_results(results_file: Path) -> list[float]:
    """Run the vendored upstream convert_results.py and parse its 'Cut Paste Results' line (1 decimal)."""
    from teval.utils.convert_results import convert_results
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        convert_results(str(results_file))
    line = next(l for l in buffer.getvalue().splitlines() if l.startswith("Cut Paste Results"))
    return [float(x) if x != "nan" else math.nan for x in line.split("[", 1)[1].rstrip("]").split()]


def close(a, b, tol=1e-6) -> bool:
    if a is None or b is None:
        return a is b
    if math.isnan(a) or math.isnan(b):
        return math.isnan(a) and math.isnan(b)
    return abs(a - b) <= tol
