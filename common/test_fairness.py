"""CPU-only tests of the pair-comparison rules: python -m unittest -v common/test_fairness.py"""
import copy
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fairness  # noqa: E402

PARENT = {"repo": "Qwen/Qwen2.5-3B-Instruct", "revision": "a" * 40}
VARIANT = {"repo": "org/Qwen2.5-3B-Instruct-SparseGPT-50", "revision": "b" * 40}
OK = {"failures": [], "quantization": None, "sparsity": {"status": "measured", "zero_fraction": 0.5}}


def manifest(model, protocol=None, items="x", fair=True):
    protocol = protocol or {"max_new_tokens": 1024, "temperature": 0.0, "versions": {"vllm": "0.10.0"}}
    return {"status": "complete", "benchmark": "demo", "protocol": protocol,
            "protocol_sha256": fairness.protocol_sha256(protocol), "model": model, "parent": PARENT,
            "checkpoint_checks": OK if fair else {"failures": ["parent check: mismatch"]},
            "fair_checkpoint": fair, "items_sha256": items, "n_items": 10, "scores": {"acc": 1.0}}


class CompareManifests(unittest.TestCase):
    def test_fair_pair(self):
        self.assertTrue(fairness.compare_manifests(manifest(PARENT), manifest(VARIANT))["fair"])

    def test_protocol_difference_fails(self):
        variant = manifest(VARIANT, {"max_new_tokens": 4096, "temperature": 0.0, "versions": {}})
        result = fairness.compare_manifests(manifest(PARENT), variant)
        self.assertFalse(result["fair"])
        self.assertIn("max_new_tokens", result["problems"][0])

    def test_package_version_difference_fails(self):
        protocol = {"max_new_tokens": 1024, "temperature": 0.0, "versions": {"vllm": "0.11.0"}}
        result = fairness.compare_manifests(manifest(PARENT), manifest(VARIANT, protocol))
        self.assertFalse(result["fair"])
        self.assertEqual(result["package_version_differences"], ["vllm"])

    def test_item_difference_fails(self):
        self.assertFalse(fairness.compare_manifests(manifest(PARENT), manifest(VARIANT, items="y"))["fair"])

    def test_wrong_parent_checkpoint_fails(self):
        self.assertFalse(fairness.compare_manifests(manifest(PARENT), manifest(VARIANT, fair=False))["fair"])

    def test_baseline_must_be_parent(self):
        self.assertFalse(fairness.compare_manifests(manifest(VARIANT), manifest(VARIANT))["fair"])

    def test_different_parents_fail(self):
        variant = manifest(VARIANT)
        variant["parent"] = {"repo": "Qwen/Qwen2.5-3B", "revision": "c" * 40}
        self.assertFalse(fairness.compare_manifests(manifest(PARENT), variant)["fair"])

    def test_smoke_runs_never_pair_with_full_runs(self):
        smoke = copy.deepcopy(manifest(VARIANT))
        smoke["protocol"]["smoke"] = 5
        self.assertFalse(fairness.compare_manifests(manifest(PARENT), smoke)["fair"])

    def test_identity_requires_pinned_revision(self):
        with self.assertRaises(ValueError):
            fairness.identity("Qwen/Qwen2.5-3B-Instruct", None)
        with self.assertRaises(ValueError):
            fairness.identity("Qwen/Qwen2.5-3B-Instruct", "main")


if __name__ == "__main__":
    unittest.main()
