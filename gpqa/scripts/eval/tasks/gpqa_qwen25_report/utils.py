"""GPQA helpers for the Qwen2.5 technical-report protocol.

Prompt, choice permutation and the strict answer regex follow OpenAI
simple-evals (gpqa_eval.py). The permutation uses its own seeded RNG, so the
correct letter does not depend on lm-eval's global random state.
"""

from __future__ import annotations

import random
import re

import datasets

from lm_eval.api.filter import Filter


QUERY_TEMPLATE = (
    "Answer the following multiple choice question. The last line of your "
    "response should be of the following format: 'Answer: $LETTER' (without "
    "quotes) where LETTER is one of ABCD. Think step by step before answering.\n"
    "\n"
    "{question}\n"
    "\n"
    "A) {A}\n"
    "B) {B}\n"
    "C) {C}\n"
    "D) {D}"
)
PERMUTATION_SEED = 0

_SIMPLE_EVALS_RE = re.compile(r"(?i)Answer[ \t]*:[ \t]*\$?([A-D])\$?")
_BOXED_RE = re.compile(r"\\boxed\s*\{\s*(?:\\text\{)?\s*\(?([A-D])\)?", re.IGNORECASE)
_THE_ANSWER_IS_RE = re.compile(r"answer\s+is\s*:?\s*\**\(?([A-D])\)?\b", re.IGNORECASE)
_PAREN_LETTER_RE = re.compile(r"\(([A-D])\)")


def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    rng = random.Random(PERMUTATION_SEED)
    permutations = [rng.sample(range(4), 4) for _ in range(len(dataset))]

    def _process(doc, idx):
        options = [
            doc["Correct Answer"].strip(),
            doc["Incorrect Answer 1"].strip(),
            doc["Incorrect Answer 2"].strip(),
            doc["Incorrect Answer 3"].strip(),
        ]
        perm = permutations[idx]
        choices = [options[i] for i in perm]
        return {
            "question": doc["Question"].strip(),
            "choices": choices,
            "answer": "ABCD"[perm.index(0)],
        }

    return dataset.map(_process, with_indices=True)


def doc_to_text(doc) -> str:
    a, b, c, d = doc["choices"]
    return QUERY_TEMPLATE.format(question=doc["question"], A=a, B=b, C=c, D=d)


def _post_think(text: str) -> str:
    return text.split("</think>")[-1] if "</think>" in text else text


def extract_simple_evals(text: str) -> str | None:
    match = _SIMPLE_EVALS_RE.search(_post_think(text or ""))
    return match.group(1).upper() if match else None


def extract_flexible(text: str) -> str | None:
    tail = _post_think(text or "")
    for pattern in (_SIMPLE_EVALS_RE, _BOXED_RE, _THE_ANSWER_IS_RE, _PAREN_LETTER_RE):
        matches = pattern.findall(tail)
        if matches:
            return matches[-1].upper()
    return None


class _LetterFilter(Filter):
    extract = staticmethod(extract_simple_evals)

    def apply(self, resps, docs):
        return [[self.extract(r) or "[invalid]" for r in inst] for inst in resps]


class SimpleEvalsFilter(_LetterFilter):
    extract = staticmethod(extract_simple_evals)


class FlexibleFilter(_LetterFilter):
    extract = staticmethod(extract_flexible)
