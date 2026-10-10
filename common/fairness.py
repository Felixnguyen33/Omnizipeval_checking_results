"""Shared fairness checks for every OmniZipEval benchmark runner.

A comparison between a dense model and a compressed variant is only fair when:

1. both runs use the same protocol (prompt, shots, decoding, token budget,
   context, dataset revision, split, scorer, judge, seeds, backend);
2. the variant was produced from that same dense parent (not, e.g., from the
   base model when the baseline is the Instruct model);
3. the variant is what its label says (actual bit width, actual sparsity).

Runners put every protocol constant in code and pass it to ``Manifest``.
``Manifest`` fingerprints it, records model identity, and runs ``checkpoint_report``,
which checks (2) and (3) before any GPU time is spent.
``compare_manifests`` is used by every ``check_pair.py`` to refuse unequal protocols.

Why the parent check uses the token embedding: none of the six methods in the paper
(Wanda, SparseGPT, GBLM-Pruner, GPTQ, AWQ, SmoothQuant) changes ``embed_tokens``.
A variant pruned or quantized from the right parent therefore has the parent's
embedding table, up to a dtype cast. Base and Instruct checkpoints of the same family
have different embedding tables, so a wrong parent is detected.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import re
import subprocess
import time
from pathlib import Path

# The checkpoint checks below run torch CPU ops in this process. vLLM's default "fork" worker start
# then deadlocks in OpenMP on its first CPU op, so ask vLLM to spawn workers instead. This does not
# change any output. Runners that set the variable themselves keep their value.
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

PROTOCOL_EXEMPT_KEYS = ()  # every protocol key must match, including package versions (they change outputs)


# ----------------------------------------------------------------------------- identity

def identity(value: str, revision: str | None) -> dict:
    """Identify a checkpoint: a local directory, or a Hub id pinned to a full commit SHA."""
    path = Path(value)
    if path.is_dir():
        if revision:
            raise ValueError(f"{value}: do not give a Hub revision for a local directory")
        config = path / "config.json"
        return {"path": str(path.resolve()),
                "config_sha256": hashlib.sha256(config.read_bytes()).hexdigest() if config.exists() else None}
    if not revision or not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError(f"{value}: Hub models require --revision / --parent-revision as a 40-character commit SHA")
    return {"repo": value, "revision": revision}


def _local_file(model: str, revision: str | None, filename: str) -> Path | None:
    path = Path(model)
    if path.is_dir():
        candidate = path / filename
        return candidate if candidate.exists() else None
    from huggingface_hub import hf_hub_download
    from huggingface_hub.utils import EntryNotFoundError
    try:
        return Path(hf_hub_download(model, filename, revision=revision))
    except EntryNotFoundError:
        return None


local_file = _local_file  # public name: resolve one file of a local directory or pinned Hub checkpoint


def read_config(model: str, revision: str | None) -> dict:
    config = _local_file(model, revision, "config.json")
    if config is None:
        raise FileNotFoundError(f"{model}: no config.json")
    return json.loads(config.read_text())


def quantization_summary(config: dict) -> dict | None:
    """Return the declared quantization (method, bits, group size), including nested VLM text configs."""
    q = config.get("quantization_config") or (config.get("text_config") or {}).get("quantization_config")
    if not q:
        return None
    summary = {"quant_method": q.get("quant_method"), "bits": q.get("bits"), "group_size": q.get("group_size")}
    groups = q.get("config_groups") or {}
    for name, group in groups.items():  # compressed-tensors (llm-compressor) format
        weights = group.get("weights") or {}
        acts = group.get("input_activations") or {}
        summary.setdefault("groups", {})[name] = {
            "weight_bits": weights.get("num_bits"), "weight_strategy": weights.get("strategy"),
            "group_size": weights.get("group_size"), "activation_bits": acts.get("num_bits") if acts else None,
            "activation_dynamic": acts.get("dynamic") if acts else None}
    return summary


def effective_bits(summary: dict | None) -> set[int]:
    if not summary:
        return set()
    bits = {summary["bits"]} if summary.get("bits") else set()
    bits |= {g["weight_bits"] for g in summary.get("groups", {}).values() if g.get("weight_bits")}
    return bits


# ----------------------------------------------------------------------------- tensors

def _weight_index(model: str, revision: str | None) -> dict[str, str]:
    index = _local_file(model, revision, "model.safetensors.index.json")
    if index is not None:
        return json.loads(index.read_text())["weight_map"]
    single = _local_file(model, revision, "model.safetensors")
    if single is None:
        raise FileNotFoundError(f"{model}: no safetensors weights (.bin checkpoints are not supported)")
    from safetensors import safe_open
    with safe_open(str(single), framework="pt") as f:
        return {k: "model.safetensors" for k in f.keys()}


def _load_tensor(model: str, revision: str | None, key: str, index: dict[str, str]):
    from safetensors import safe_open
    shard = _local_file(model, revision, index[key])
    with safe_open(str(shard), framework="pt") as f:
        return f.get_tensor(key)


def _embedding_key(index: dict[str, str]) -> str | None:
    keys = sorted(k for k in index if k.endswith("embed_tokens.weight"))
    # VLMs may also contain a vision embedding; prefer the language-model table (the largest name match).
    keys.sort(key=lambda k: ("language_model" not in k and "model.embed_tokens" not in k, len(k)))
    return keys[0] if keys else None


def parent_check(model: str, revision: str | None, parent: str, parent_revision: str | None) -> dict:
    """Compare the token-embedding tables of a variant and its claimed dense parent."""
    import torch
    torch.set_num_threads(1)  # keep OpenMP single-threaded before any later fork (see top of file)
    m_index, p_index = _weight_index(model, revision), _weight_index(parent, parent_revision)
    m_key, p_key = _embedding_key(m_index), _embedding_key(p_index)
    if m_key is None or p_key is None:
        return {"status": "unverifiable", "reason": "embed_tokens.weight not found"}
    a = _load_tensor(model, revision, m_key, m_index)
    b = _load_tensor(parent, parent_revision, p_key, p_index)
    if a.shape != b.shape:
        return {"status": "mismatch", "reason": f"shape {tuple(a.shape)} vs {tuple(b.shape)}",
                "variant_key": m_key, "parent_key": p_key}
    # Cast the parent to the variant's storage dtype, then compare exactly. A correct
    # parent survives the cast bit-for-bit; a different checkpoint does not.
    b_cast = b.to(a.dtype)
    equal = (a == b_cast).float().mean().item()
    max_abs = (a.float() - b.float()).abs().max().item()
    status = "match" if equal >= 0.999 else "mismatch"
    return {"status": status, "equal_fraction": round(equal, 6),
            "reason": "" if status == "match" else
            f"only {equal:.1%} of embedding entries equal the parent's; the variant was compressed from another checkpoint",
            "max_abs_diff": max_abs, "variant_key": m_key, "parent_key": p_key,
            "variant_dtype": str(a.dtype), "parent_dtype": str(b.dtype)}


# Decoder linear weights, including MoE experts (mlp.experts.N.*_proj); vision-tower layers are excluded below.
_LINEAR = re.compile(r"layers\.(\d+)\.(self_attn\.(q|k|v|o)_proj|mlp\.(experts\.\d+\.)?(gate|up|down)_proj)\.weight$")


def sampled_sparsity(model: str, revision: str | None) -> dict:
    """Zero fraction of the decoder linear weights in the first, middle and last layer.

    Only meaningful for unquantized (dense-storage) checkpoints; packed quantized weights are skipped.
    """
    import torch
    torch.set_num_threads(1)
    index = _weight_index(model, revision)
    linear = {k: int(_LINEAR.search(k).group(1)) for k in index
              if _LINEAR.search(k) and "vision" not in k and "visual" not in k}
    if not linear:
        return {"status": "unmeasurable", "reason": "no dense decoder linear weights (packed or quantized format)"}
    layers = sorted(set(linear.values()))
    picked = {layers[0], layers[len(layers) // 2], layers[-1]}
    zeros = total = 0
    per_layer = {}
    for key, layer in sorted(linear.items()):
        if layer not in picked:
            continue
        t = _load_tensor(model, revision, key, index)
        if t.dtype in (torch.int8, torch.int32, torch.uint8):
            return {"status": "unmeasurable", "reason": f"{key} is stored as {t.dtype}"}
        z = int((t == 0).sum().item())
        zeros, total = zeros + z, total + t.numel()
        per_layer.setdefault(layer, [0, 0])
        per_layer[layer][0] += z
        per_layer[layer][1] += t.numel()
    return {"status": "measured", "zero_fraction": round(zeros / total, 4),
            "per_layer": {str(k): round(v[0] / v[1], 4) for k, v in sorted(per_layer.items())},
            "layers_sampled": sorted(picked)}


def checkpoint_report(model: str, revision: str | None, parent: str, parent_revision: str | None,
                      expect: dict | None = None) -> dict:
    """Declared quantization, sampled sparsity and parent-embedding check for one checkpoint.

    ``expect`` may hold {"bits": 4} or {"sparsity": 0.5}; a contradiction is reported as a failure.
    """
    config = read_config(model, revision)
    summary = quantization_summary(config)
    report = {"architectures": config.get("architectures"), "name_or_path": config.get("_name_or_path"),
              "quantization": summary, "parent": parent_check(model, revision, parent, parent_revision)}
    report["sparsity"] = sampled_sparsity(model, revision) if summary is None else {
        "status": "unmeasurable", "reason": "quantized checkpoint"}
    failures = []
    if report["parent"]["status"] != "match":
        failures.append(f"parent check: {report['parent']['status']} ({report['parent'].get('reason', '')})")
    expect = expect or {}
    if "bits" in expect and expect["bits"] not in effective_bits(summary):
        failures.append(f"expected {expect['bits']}-bit weights, config declares {sorted(effective_bits(summary)) or 'none'}")
    if "sparsity" in expect:
        s = report["sparsity"]
        if s["status"] != "measured" or abs(s["zero_fraction"] - expect["sparsity"]) > 0.02:
            failures.append(f"expected sparsity {expect['sparsity']}, measured {s.get('zero_fraction', s['status'])}")
    report["failures"] = failures
    return report


# ----------------------------------------------------------------------------- manifest

def protocol_sha256(protocol: dict) -> str:
    core = {k: v for k, v in protocol.items() if k not in PROTOCOL_EXEMPT_KEYS}
    return hashlib.sha256(json.dumps(core, sort_keys=True, default=str).encode()).hexdigest()


def package_versions(*names: str) -> dict:
    out = {}
    for name in names:
        try:
            out[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            out[name] = None
    return out


def source_hashes(*paths: Path) -> dict:
    return {str(p.name): hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in paths}


def repo_commit() -> str | None:
    try:
        root = Path(__file__).resolve().parent.parent
        commit = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True,
                                check=True).stdout.strip()
        dirty = subprocess.run(["git", "-C", str(root), "status", "--porcelain"], capture_output=True, text=True,
                               check=True).stdout.strip()
        return commit + ("-dirty" if dirty else "")
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


class Manifest:
    """``manifest.json`` for one run: protocol, model identity and checkpoint checks.

    The output directory must not exist yet (no silent overwrite or resume).
    """

    def __init__(self, output: Path, benchmark: str, protocol: dict, model: str, revision: str | None,
                 parent: str, parent_revision: str | None, expect: dict | None = None,
                 allow_unverified_parent: bool = False, skip_checkpoint_checks: bool = False):
        output = Path(output)
        if output.exists():
            raise SystemExit(f"{output} already exists; choose a new output directory")
        self.output = output
        model_id, parent_id = identity(model, revision), identity(parent, parent_revision)
        if skip_checkpoint_checks:
            checks = {"skipped": True, "failures": ["checkpoint checks skipped"]}
        else:
            checks = checkpoint_report(model, revision, parent, parent_revision, expect)
        if checks["failures"] and not allow_unverified_parent:
            raise SystemExit("Refusing an unfair run:\n  " + "\n  ".join(checks["failures"]) +
                             "\nFix the checkpoint, or pass --allow-unverified-parent (recorded in the manifest).")
        output.mkdir(parents=True)
        self.data = {"status": "started", "benchmark": benchmark, "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
                     "repo_commit": repo_commit(), "protocol": protocol,
                     "protocol_sha256": protocol_sha256(protocol), "model": model_id, "parent": parent_id,
                     "expect": expect or {}, "checkpoint_checks": checks,
                     "fair_checkpoint": not checks["failures"]}
        self.write()

    def write(self):
        (self.output / "manifest.json").write_text(json.dumps(self.data, indent=2, default=str) + "\n")

    def complete(self, scores: dict, item_ids: list):
        self.data.update(status="complete", finished=time.strftime("%Y-%m-%dT%H:%M:%S"), scores=scores,
                         n_items=len(item_ids),
                         items_sha256=hashlib.sha256(json.dumps(sorted(map(str, item_ids))).encode()).hexdigest())
        (self.output / "items.json").write_text(json.dumps(sorted(map(str, item_ids))) + "\n")
        self.write()


def load_manifest(folder: Path) -> dict:
    manifest = json.loads((Path(folder) / "manifest.json").read_text())
    if manifest.get("status") != "complete":
        raise SystemExit(f"{folder}: run is not complete")
    return manifest


def compare_manifests(baseline: dict, variant: dict) -> dict:
    """Return everything that makes a baseline/variant pair unfair (empty lists mean fair)."""
    problems = []
    if baseline["benchmark"] != variant["benchmark"]:
        problems.append(f"benchmarks differ: {baseline['benchmark']} vs {variant['benchmark']}")
    keys = set(baseline["protocol"]) | set(variant["protocol"])
    protocol_diff = sorted(k for k in keys if k not in PROTOCOL_EXEMPT_KEYS and k != "versions"
                           and baseline["protocol"].get(k) != variant["protocol"].get(k))
    if protocol_diff:
        problems.append("protocol differs in: " + ", ".join(protocol_diff))
    if baseline.get("items_sha256") != variant.get("items_sha256"):
        problems.append(f"evaluated items differ ({baseline.get('n_items')} vs {variant.get('n_items')})")
    if baseline["parent"] != variant["parent"]:
        problems.append("the two runs name different dense parents")
    if baseline["model"] != baseline["parent"]:
        problems.append("the baseline is not the dense parent itself")
    for name, m in (("baseline", baseline), ("variant", variant)):
        if not m.get("fair_checkpoint"):
            problems.append(f"{name} checkpoint checks failed: {m['checkpoint_checks'].get('failures')}")
    version_diff = sorted(k for k in set(baseline["protocol"].get("versions", {})) |
                          set(variant["protocol"].get("versions", {}))
                          if baseline["protocol"].get("versions", {}).get(k) !=
                          variant["protocol"].get("versions", {}).get(k))
    if version_diff:
        problems.append("package versions differ (rerun both in one environment): " + ", ".join(version_diff))
    return {"fair": not problems, "problems": problems, "package_version_differences": version_diff,
            "baseline_scores": baseline.get("scores"), "variant_scores": variant.get("scores"),
            "variant_quantization": variant["checkpoint_checks"].get("quantization"),
            "variant_sparsity": variant["checkpoint_checks"].get("sparsity")}


def add_model_arguments(parser):
    """CLI arguments shared by every runner. Protocol settings are deliberately not CLI options."""
    parser.add_argument("--model", required=True, help="Hub id or local directory of the model to evaluate")
    parser.add_argument("--revision", help="40-character Hub commit of --model (required for Hub ids)")
    parser.add_argument("--parent", required=True,
                        help="Dense parent the variant was compressed from; for the dense run pass the same as --model. "
                             "Its tokenizer, chat template and processor are used for every run in the cohort.")
    parser.add_argument("--parent-revision", help="40-character Hub commit of --parent (required for Hub ids)")
    parser.add_argument("--output", required=True, type=Path, help="New output directory (refused if it exists)")
    parser.add_argument("--expect-bits", type=int, help="Fail unless the config declares this weight bit width")
    parser.add_argument("--expect-sparsity", type=float, help="Fail unless sampled sparsity is within 0.02 of this")
    parser.add_argument("--allow-unverified-parent", action="store_true",
                        help="Run even if checkpoint checks fail; the manifest marks the run unfair")
    parser.add_argument("--skip-checkpoint-checks", action="store_true",
                        help="Do not download weights for the checks; the manifest marks the run unfair")
    return parser


def expect_from_args(args) -> dict:
    expect = {}
    if args.expect_bits is not None:
        expect["bits"] = args.expect_bits
    if args.expect_sparsity is not None:
        expect["sparsity"] = args.expect_sparsity
    return expect


def manifest_from_args(args, benchmark: str, protocol: dict) -> Manifest:
    return Manifest(args.output, benchmark, protocol, args.model, args.revision, args.parent, args.parent_revision,
                    expect_from_args(args), args.allow_unverified_parent, args.skip_checkpoint_checks)
