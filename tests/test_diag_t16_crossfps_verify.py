"""Hardened-verifier tests for DIAG_T16_CROSSFPS (stdlib unittest; no pytest).

Run:
  .venv/bin/python -m unittest tests.test_diag_t16_crossfps_verify -v
"""
import json
import sys
import unittest
from pathlib import Path

import numpy as np

WS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WS / "deepfake" / "bench"))
sys.path.insert(0, str(WS / "deepfake" / "scripts"))

import verify_diag_t16_crossfps as vf  # noqa: E402

OUT = WS / "artifacts" / "split_t2v_matched" / "diag_t16_crossfps"


def _row(arm="genvid_fake", gen="g", fn="f.mp4", sha="a" * 64,
         label=1, score=0.5, recipe="r"):
    return {"arm": arm, "generator": gen, "filename": fn, "sha256": sha,
            "path": f"/x/{fn}", "label": label, "score": score,
            "recipe_hash": recipe}


class TestAuc(unittest.TestCase):
    def test_matches_sklearn_with_ties(self):
        from sklearn.metrics import roc_auc_score
        rng = np.random.default_rng(0)
        f = rng.integers(0, 5, 40).astype(float)
        r = rng.integers(0, 5, 60).astype(float)
        y = [1] * len(f) + [0] * len(r)
        self.assertAlmostEqual(vf.mann_whitney_auc(f, r),
                               roc_auc_score(y, np.concatenate([f, r])),
                               places=12)

    def test_perfect_and_inverted(self):
        self.assertEqual(vf.mann_whitney_auc([0.9, 0.8], [0.1, 0.2]), 1.0)
        self.assertEqual(vf.mann_whitney_auc([0.1, 0.2], [0.9, 0.8]), 0.0)

    def test_all_ties_is_half(self):
        self.assertEqual(vf.mann_whitney_auc([1.0, 1.0], [1.0, 1.0]), 0.5)

    def test_empty_arm_raises(self):
        with self.assertRaises(ValueError):
            vf.mann_whitney_auc([], [1.0])


class TestCheckRows(unittest.TestCase):
    def setUp(self):
        self.doc = {"recipe_hash": "r",
                    "rows": [_row(sha="a" * 64), _row(fn="g.mp4", sha="b" * 64)]}
        self.scores = [dict(r) for r in self.doc["rows"]]

    def test_clean(self):
        self.assertEqual(vf.check_rows(self.doc, self.scores), [])

    def test_same_key_payload_mismatch_detected(self):
        self.scores[0]["sha256"] = "c" * 64
        probs = vf.check_rows(self.doc, self.scores)
        self.assertTrue(any("payload mismatch" in p for p in probs))

    def test_same_key_label_mismatch_detected(self):
        self.scores[0]["label"] = 0
        self.assertTrue(any("payload mismatch" in p
                            for p in vf.check_rows(self.doc, self.scores)))

    def test_path_mismatch_detected(self):
        self.scores[0]["path"] = "/y/f.mp4"
        self.assertTrue(any("payload mismatch" in p
                            for p in vf.check_rows(self.doc, self.scores)))

    def test_score_recipe_mismatch_detected(self):
        self.scores[0]["recipe_hash"] = "other"
        self.assertTrue(any("score recipe mismatch" in p
                            for p in vf.check_rows(self.doc, self.scores)))

    def test_duplicate_key_detected(self):
        self.scores.append(dict(self.scores[0]))
        self.assertTrue(any("duplicate" in p
                            for p in vf.check_rows(self.doc, self.scores)))

    def test_foreign_and_missing_detected(self):
        self.scores[0]["filename"] = "zzz.mp4"
        probs = vf.check_rows(self.doc, self.scores)
        self.assertTrue(any("row set mismatch" in p for p in probs))


class TestCheckNegatives(unittest.TestCase):
    def _neg(self, n=1000, label=0, score=0.5, recipe="r"):
        return [{"arm": "stage0_real", "generator": "g", "filename": f"{i}.mp4",
                 "label": label, "score": score, "recipe_hash": recipe}
                for i in range(n)]

    def test_clean(self):
        self.assertEqual(vf.check_negatives(self._neg(), "r"), [])

    def test_wrong_count_detected(self):
        probs = vf.check_negatives(self._neg(n=999), "r")
        self.assertTrue(any("n=999" in p for p in probs))

    def test_wrong_label_detected(self):
        self.assertTrue(any("label" in p
                            for p in vf.check_negatives(self._neg(label=1), "r")))

    def test_unavailable_detected(self):
        self.assertTrue(any("unavailable" in p
                            for p in vf.check_negatives(self._neg(score=None), "r")))

    def test_recipe_drift_detected(self):
        self.assertTrue(any("recipe" in p
                            for p in vf.check_negatives(self._neg(recipe="x"), "r")))


class TestStage0Recipe(unittest.TestCase):
    def test_match(self):
        s0 = vf.load_seal_any(vf.STAGE0_DIR / "seal.txt")
        self.assertEqual(vf.check_stage0_recipe({"recipe_hash": s0["recipe_hash"]}), [])

    def test_mismatch_detected(self):
        self.assertTrue(vf.check_stage0_recipe({"recipe_hash": "0" * 64}))


class TestInputHashes(unittest.TestCase):
    def test_missing_input_detected(self):
        seal = {"stage0_scores": "0" * 64 + " /nope/scores.jsonl"}
        self.assertTrue(any("missing" in p for p in vf.check_input_hashes(seal)))

    def test_unrecorded_input_detected(self):
        self.assertTrue(vf.check_input_hashes({}))


class TestIntegration(unittest.TestCase):
    def test_real_artifacts_verify(self):
        report, problems = vf.verify(OUT)
        self.assertEqual(problems, [])
        self.assertTrue(report["ok"])
        self.assertEqual(report["n_fakes"], 434)
        self.assertEqual(report["n_negatives"], 1000)
        self.assertAlmostEqual(report["primary_auc_recomputed"], 0.5398594470046083,
                               delta=1e-9)
        self.assertAlmostEqual(report["primary_auc_dedup"], 0.538799,
                               delta=1e-6)
        self.assertLess(report["primary_auc_recomputed"], 0.70)
        self.assertEqual(report["band"], "NO_TRANSFER_EVIDENCE")
        self.assertEqual(report["checks"]["booked"], 0)


if __name__ == "__main__":
    unittest.main()
