"""DIAG_T16_CROSSFPS pre-registration tests (stdlib unittest; no pytest).

Run:
  .venv/bin/python -m unittest tests.test_diag_t16_crossfps -v
"""
import json
import sys
import tempfile
import unittest
from pathlib import Path

WS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WS / "deepfake" / "bench"))
sys.path.insert(0, str(WS / "deepfake" / "scripts"))

import diag_t16_crossfps as dg  # noqa: E402


def _fake(score):
    return {"arm": "genvid_fake", "generator": "g", "filename": "f.mp4",
            "label": 1, "score": score, "fps": 30.0}


def _real(score):
    return {"arm": "stage0_real", "generator": "g", "filename": "r.mp4",
            "label": 0, "score": score, "fps": 24.0}


class TestAuc(unittest.TestCase):
    def test_perfect_separation(self):
        fakes = [_fake(s) for s in (0.9, 0.8, 0.7)]
        reals = [_real(s) for s in (0.1, 0.2, 0.3)]
        self.assertEqual(dg._auc(fakes, reals), 1.0)

    def test_inverted_separation(self):
        fakes = [_fake(s) for s in (0.1, 0.2, 0.3)]
        reals = [_real(s) for s in (0.9, 0.8, 0.7)]
        self.assertEqual(dg._auc(fakes, reals), 0.0)

    def test_all_ties_is_half(self):
        fakes = [_fake(0.5), _fake(0.5)]
        reals = [_real(0.5), _real(0.5)]
        self.assertEqual(dg._auc(fakes, reals), 0.5)

    def test_none_scores_are_dropped_not_zeroed(self):
        fakes = [_fake(0.9), _fake(None)]
        reals = [_real(0.1), _real(None)]
        self.assertEqual(dg._auc(fakes, reals), 1.0)

    def test_empty_arm_is_none(self):
        self.assertIsNone(dg._auc([_fake(None)], [_real(0.1)]))


class TestBand(unittest.TestCase):
    def test_boundaries(self):
        self.assertEqual(dg._band(0.85)[0], "GATE_HAS_SIGNAL")
        self.assertEqual(dg._band(0.8499)[0], "PARTIAL")
        self.assertEqual(dg._band(0.70)[0], "PARTIAL")
        self.assertEqual(dg._band(0.6999)[0], "NO_TRANSFER_EVIDENCE")
        self.assertEqual(dg._band(None)[0], "INCONCLUSIVE")

    def test_action_text_is_precommitted(self):
        self.assertIn("justifies Stage 1", dg._band(0.90)[1])
        self.assertIn("closing the SPLIT line", dg._band(0.50)[1])


class TestScoreLoading(unittest.TestCase):
    def test_duplicate_keys_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "scores.jsonl"
            row = {**_fake(0.5), "recipe_hash": "x"}
            p.write_text(json.dumps(row) + "\n" + json.dumps(row) + "\n")
            self.assertEqual(dg._load_scores(p), [])

    def test_corrupt_line_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "scores.jsonl"
            p.write_text('{"arm": "x"}\nnot json\n')
            self.assertEqual(dg._load_scores(p), [])

    def test_clean_file_loads(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "scores.jsonl"
            p.write_text(json.dumps(_fake(0.5)) + "\n")
            self.assertEqual(len(dg._load_scores(p)), 1)


class TestSeal(unittest.TestCase):
    def _sealed_dir(self, td):
        out = Path(td)
        rows = {"stage": "diag_t16_crossfps", "n_rows": 2, "rows": [],
                "inputs": {}}
        (out / "rows.json").write_text(json.dumps(rows) + "\n")
        self.assertEqual(dg.write_seal(out), 0)
        return out

    def test_clean_seal_has_no_problems(self):
        with tempfile.TemporaryDirectory() as td:
            out = self._sealed_dir(td)
            self.assertEqual(dg.seal_problems(out), [])

    def test_rows_tamper_detected(self):
        with tempfile.TemporaryDirectory() as td:
            out = self._sealed_dir(td)
            with open(out / "rows.json", "a") as f:
                f.write(" ")
            self.assertIn("rows.json changed since seal",
                          dg.seal_problems(out))

    def test_module_tamper_detected(self):
        with tempfile.TemporaryDirectory() as td:
            out = self._sealed_dir(td)
            lines = (out / "seal.txt").read_text().splitlines()
            lines = ["module " + "0" * 64 if l.startswith("module ") else l
                     for l in lines]
            (out / "seal.txt").write_text("\n".join(lines) + "\n")
            self.assertIn("module changed since seal", dg.seal_problems(out))

    def test_missing_seal_detected(self):
        with tempfile.TemporaryDirectory() as td:
            self.assertEqual(dg.seal_problems(Path(td)), ["seal.txt missing"])

    def test_foreign_row_blocks_verdict(self):
        with tempfile.TemporaryDirectory() as td:
            out = self._sealed_dir(td)
            foreign = {"arm": "genvid_fake", "generator": "g",
                       "filename": "not_bound.mp4", "label": 1,
                       "score": 0.5, "recipe_hash": dg.st.split_recipe_hash()}
            (out / "scores.jsonl").write_text(json.dumps(foreign) + "\n")
            self.assertEqual(dg.verdict(out), 2)


class TestSpearman(unittest.TestCase):
    def test_constant_fps_returns_none(self):
        rows = [{**_fake(0.1), "fps": 24.0}, {**_fake(0.2), "fps": 24.0},
                {**_fake(0.3), "fps": 24.0}]
        self.assertIsNone(dg._spearman(rows)["rho"])

    def test_monotone_fps_correlation(self):
        rows = [{**_fake(0.1), "fps": 8.0}, {**_fake(0.2), "fps": 16.0},
                {**_fake(0.3), "fps": 24.0}]
        self.assertAlmostEqual(dg._spearman(rows)["rho"], 1.0, places=6)


class TestNoProductionSurface(unittest.TestCase):
    def test_no_production_imports(self):
        src = (WS / "deepfake" / "scripts" / "diag_t16_crossfps.py").read_text()
        for bad in ("argus_detect", "video_channels", "server",
                    "split_temporal", "exam_split_matched"):
            self.assertNotIn(f"import {bad}", src)
            self.assertNotIn(f"from {bad}", src)

    def test_does_not_edit_sealed_module(self):
        src = (WS / "deepfake" / "scripts" / "diag_t16_crossfps.py").read_text()
        self.assertIn("read-only", src)
        self.assertNotIn("st.score_clip = ", src)


if __name__ == "__main__":
    unittest.main()
