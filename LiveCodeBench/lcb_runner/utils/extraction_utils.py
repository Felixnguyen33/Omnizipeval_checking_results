import re

from lcb_runner.lm_styles import LMStyle


_THINK_CLOSE = "</think>"
_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)


def repair_raw_sentencepiece_text(text: str) -> str:
    """Repair outputs where vLLM joined SP/BPE piece names instead of decoding.

    Symptom: literal ``Ġ`` (space) / ``Ċ`` (newline) markers in the string.
    """
    if not text:
        return text
    if text.count("Ġ") < 3 and text.count("Ċ") < 3:
        return text
    repaired = (
        text.replace("Ċ", "\n")
        .replace("ĉ", "\t")
        .replace("Ġ", " ")
        .replace("âĨĴ", "→")
    )
    return repaired


def _strip_thinking(model_output: str) -> str:
    if _THINK_CLOSE in model_output:
        return model_output.split(_THINK_CLOSE)[-1].strip()
    return model_output.strip()


def _extract_fenced_code(text: str) -> str:
    if not text:
        return ""
    outputlines = text.split("\n")
    py_starts = [
        i
        for i, line in enumerate(outputlines)
        if "```python" in line or "```Python" in line
    ]
    if py_starts:
        start = py_starts[-1]
        for j in range(start + 1, len(outputlines)):
            if outputlines[j].strip().startswith("```"):
                return "\n".join(outputlines[start + 1 : j])
    indexlines = [i for i, line in enumerate(outputlines) if "```" in line]
    if len(indexlines) < 2:
        return ""
    return "\n".join(outputlines[indexlines[-2] + 1 : indexlines[-1]])


def _looks_like_python(code: str) -> bool:
    s = code.strip()
    if not s or s == "# YOUR CODE HERE":
        return False
    markers = ("def ", "class ", "import ", "from ", "print(", "input(", "sys.")
    return any(m in s for m in markers)


def extract_code(model_output: str, lmstyle: LMStyle):
    model_output = repair_raw_sentencepiece_text(model_output or "")
    outputlines = model_output.split("\n")

    if lmstyle == LMStyle.CodeLLaMaInstruct:
        indexlines = [i for i, line in enumerate(outputlines) if "PYTHON]" in line]
        if len(indexlines) < 2:
            indexlines = [i for i, line in enumerate(outputlines) if "```" in line]
        if len(indexlines) < 2:
            return ""
        return "\n".join(outputlines[indexlines[-2] + 1 : indexlines[-1]])

    if lmstyle == LMStyle.GenericBase:
        return model_output.strip()

    # Reasoning models (R1 / Qwen3 / etc.): prefer post-thinking answer.
    is_thinking = (
        lmstyle == LMStyle.DeepSeekR1
        or "<think>" in model_output
        or _THINK_CLOSE in model_output
    )
    if is_thinking:
        after = _strip_thinking(model_output)
        answer = _ANSWER_RE.search(after)
        if answer:
            body = answer.group(1).strip()
            fenced = _extract_fenced_code(body)
            if fenced.strip():
                return fenced
            if _looks_like_python(body):
                return body
        fenced = _extract_fenced_code(after)
        if fenced.strip():
            return fenced
        # Fallback: fences anywhere in the full generation
        return _extract_fenced_code(model_output)

    return _extract_fenced_code(model_output)


def extract_test_output_code(model_output: str, lmstyle: LMStyle = None):
    model_output = repair_raw_sentencepiece_text(model_output or "")
    outputlines = model_output.split("\n")
    # find the last line startwith assert...
    indexlines = [i for i, line in enumerate(outputlines) if line.startswith("assert")]
    if indexlines:
        return outputlines[indexlines[-1]]
    if lmstyle and lmstyle == LMStyle.CodeLLaMaInstruct:
        indexlines = [i for i, line in enumerate(outputlines) if "PYTHON]" in line]
    else:
        # first try to extract ```python if not then try ```
        indexlines = [
            i
            for i, line in enumerate(outputlines)
            if "```python" in line or "```Python" in line
        ]
        if indexlines:
            start_index = indexlines[0]
        else:
            start_index = None
        indexlines = [i for i, line in enumerate(outputlines) if "```" in line]
        if start_index is not None:
            indexlines = [i for i in indexlines if i > start_index]
            indexlines = [start_index] + indexlines

    if len(indexlines) < 2:
        return ""
    return "\n".join(outputlines[indexlines[0] + 1 : indexlines[1]])


def extract_execution_code(model_output: str, lmstyle: LMStyle, cot: bool = False):
    model_output = repair_raw_sentencepiece_text(model_output or "")
    if cot:
        if "[ANSWER]" in model_output:
            model_output = model_output.split("[ANSWER]")[1].strip()
    if "==" in model_output:
        model_output = model_output.split("==")[1].strip()
    if "[/ANSWER]" in model_output:
        model_output = model_output.split("[/ANSWER]")[0].strip()
    else:
        model_output = model_output.split("\n")[0].strip()
    return model_output.strip()
