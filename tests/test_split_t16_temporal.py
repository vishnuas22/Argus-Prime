"""SPLIT Stage 0 T=16 pre-registration tests (stdlib unittest; no pytest).

Run:
  .venv/bin/python -m unittest tests.test_split_t16_temporal -v
"""
import hashlib
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import cv2
import numpy as np

WS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WS / "deepfake" / "bench"))
sys.path.insert(0, str(WS / "deepfake" / "scripts"))

import split_t16_temporal as st  # noqa: E402
import exam_split_stage0_t16 as ex  # noqa: E402

GENBUSTER = WS / "data" / "raw" / "genbuster_plusplus" / "extracted" / "video"
REAL_CLIP = GENBUSTER / "real" / \
    "0025d5c07fb66035a6ae493b2f2ddcf3b9dc7889a47454914bb19645270ea0a7.mp4"
SEALED_OUT = WS / "artifacts" / "split_t2v_matched" / "stage0_t16"


class TestSplitT16Temporal(unittest.TestCase):
    def test_frame_indices_formula(self):
        self.assertEqual(st.sample_indices(24.0, 48),
                         [0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39,
                          42, 45])
        self.assertEqual(st.sample_indices(24.0, 40)[-1], 39)
        self.assertEqual(st.sample_indices(8.0, 16),
                         [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14,
                          15])

    def test_too_few_frames_fails_closed(self):
        with self.assertRaises(st.SplitUnavailable):
            st.sample_indices(24.0, 15)
        with self.assertRaises(ValueError):
            st.ttr(np.zeros((15, 4, 3), dtype=np.float32))
        with self.assertRaises(ValueError):
            st.lsmi(np.zeros((15, 4, 3), dtype=np.float32))

    def test_t_not_16_fails_closed(self):
        with self.assertRaises(ValueError):
            st.sample_indices(24.0, 48, t_frames=8)
        with self.assertRaises(ValueError):
            st.ttr(np.zeros((8, 4, 3), dtype=np.float32))
        with self.assertRaises(ValueError):
            st.lsmi(np.zeros((8, 4, 3), dtype=np.float32))
        with self.assertRaises(ValueError):
            st.preprocess_frames([np.zeros((8, 8, 3), dtype=np.uint8)] * 8)

    def test_decode_too_few_frames_video(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "seven.mp4"
            vw = cv2.VideoWriter(str(p), cv2.VideoWriter_fourcc(*"mp4v"),
                                 24.0, (64, 64))
            self.assertTrue(vw.isOpened())
            for i in range(7):
                vw.write(np.full((64, 64, 3), i * 20, dtype=np.uint8))
            vw.release()
            with self.assertRaises(st.SplitUnavailable) as ctx:
                st.decode_even_frames(p)
            self.assertIn("too_few_frames", str(ctx.exception))
            row = st.score_clip(p)
            self.assertIsNone(row["score"])
            self.assertIn("too_few_frames", row["unavailable_reason"])

    def test_duplicate_frame_indices_fails_closed(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "sixteen.mp4"
            vw = cv2.VideoWriter(str(p), cv2.VideoWriter_fourcc(*"mp4v"),
                                 24.0, (64, 64))
            self.assertTrue(vw.isOpened())
            for i in range(16):
                vw.write(np.full((64, 64, 3), i * 10, dtype=np.uint8))
            vw.release()
            row = st.score_clip(p)
            self.assertIsNone(row["score"])
            self.assertIn("duplicate_frame_indices",
                          row["unavailable_reason"])

    def test_ttr_formula(self):
        rng = np.random.default_rng(0)
        f = rng.normal(size=(16, 4, 3)).astype(np.float32)
        d1 = f[1:] - f[:-1]
        d2 = f[2:] - f[:-2]
        l1 = np.sum(np.linalg.norm(d1, axis=-1), axis=0)
        l2 = (np.sum(np.linalg.norm(d2, axis=-1), axis=0) / 2.0) \
            * ((16 - 1) / (16 - 2))
        ref = np.log2(l1 + 1e-8) - np.log2(l2 + 1e-8)
        np.testing.assert_allclose(st.ttr(f), ref, rtol=0, atol=1e-6)

    def test_lsmi_formula(self):
        rng = np.random.default_rng(1)
        f = rng.normal(size=(16, 4, 3)).astype(np.float32)
        p = f.reshape(16, 2, 2, 3)
        motion = p[1:] - p[:-1]
        gx = motion[:, :, :-1, :] - motion[:, :, 1:, :]
        gy = motion[:, :-1, :, :] - motion[:, 1:, :, :]
        ref = float((np.linalg.norm(gx, axis=-1).mean()
                     + np.linalg.norm(gy, axis=-1).mean()) / 2.0)
        self.assertAlmostEqual(st.lsmi(f), ref, places=6)

    def test_ttr_lsmi_constant_features(self):
        f = np.full((16, 4, 3), 0.25, dtype=np.float32)
        np.testing.assert_allclose(st.ttr(f), 0.0, atol=1e-6)
        self.assertEqual(st.lsmi(f), 0.0)

    def test_ttr_lsmi_uniform_translation(self):
        rng = np.random.default_rng(4)
        p0 = rng.normal(size=(4, 3)).astype(np.float32)
        v = np.array([0.1, -0.2, 0.3], dtype=np.float32)
        f = np.stack([p0 + t * v for t in range(16)]).astype(np.float32)
        np.testing.assert_allclose(st.ttr(f), 0.0, atol=1e-4)
        self.assertAlmostEqual(st.lsmi(f), 0.0, places=6)

    def test_lsmi_analytic_2x2(self):
        f = np.zeros((16, 4, 1), dtype=np.float32)
        for t in range(16):
            for i in range(2):
                for j in range(2):
                    f[t, i * 2 + j, 0] = t * (i + 2 * j)
        np.testing.assert_allclose(st.ttr(f), 0.0, atol=1e-5)
        self.assertEqual(st.lsmi(f), 1.5)

    def test_patch_shape_finite(self):
        feats = st.patch_features(REAL_CLIP)
        self.assertEqual(feats.shape, (16, 196, 768))
        self.assertEqual(feats.dtype, np.float32)
        self.assertTrue(np.isfinite(feats).all())

    def test_same_clip_twice_identical(self):
        a = st.score_clip(REAL_CLIP)
        b = st.score_clip(REAL_CLIP)
        for key in ("score", "ttr", "lsmi", "motion", "fps",
                    "n_decoded", "frame_indices"):
            self.assertEqual(a[key], b[key], key)
        self.assertEqual(a["feature_shape"], [16, 196, 768])
        self.assertEqual(a["n_frames"], 16)
        self.assertEqual(a["frame_indices"],
                         [0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39,
                          42, 45])

    def test_hash_changes_when_policy_changes(self):
        base = st.split_recipe_hash()
        self.assertEqual(base, st.split_recipe_hash())
        self.assertEqual(len(base), 64)
        self.assertEqual(st.POLICY["frames"], 16)
        self.assertEqual(st.POLICY["window_s"], 2.0)
        self.assertEqual(st.POLICY["native_num_frames"], 8)
        self.assertEqual(st.POLICY["grouping_policy"],
                         "native_two_8frame_groups")
        variants = [
            {**st.POLICY, "frames": 8},
            {**st.POLICY, "window_s": 3.0},
            {**st.POLICY, "native_num_frames": 16},
            {**st.POLICY, "gamma": 7},
            {**st.POLICY, "crop_frac_per_side": 0.05},
            {**st.POLICY, "jpeg": "q80"},
        ]
        hashes = {st.split_recipe_hash(v) for v in variants}
        self.assertEqual(len(hashes), len(variants))
        self.assertNotIn(base, hashes)

    def test_code_sha_is_this_file(self):
        p = Path(st.__file__).resolve()
        self.assertEqual(st.CODE_SHA, hashlib.sha256(p.read_bytes()).hexdigest())

    def test_motion_metric(self):
        static = [np.full((64, 64, 3), 128, dtype=np.uint8)
                  for _ in range(16)]
        self.assertEqual(st.motion_metric(static), 0.0)
        rng = np.random.default_rng(2)
        moving = [rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)
                  for _ in range(16)]
        self.assertGreater(st.motion_metric(moving), 0.0)
        with self.assertRaises(ValueError):
            st.motion_metric(static[:8])

    def test_preprocess_shape_finite(self):
        rng = np.random.default_rng(3)
        frame = rng.integers(0, 256, (1080, 1920, 3), dtype=np.uint8)
        out = st.preprocess_frame(frame)
        self.assertEqual(out.shape, (3, 224, 224))
        self.assertEqual(out.dtype, np.float32)
        self.assertTrue(np.isfinite(out).all())

    def _synth_splits(self):
        clips = []
        for arm, gen, label in (("stage0_real", "genbuster_real", 0),
                                ("stage0_fake", "genbuster_fake", 1)):
            for i in range(1000):
                clips.append({"arm": arm, "generator": gen,
                              "label": label,
                              "path": f"/tmp/{arm}/{i}.mp4",
                              "filename": f"{i:04d}.mp4",
                              "sha256": f"{arm}-{i}"})
        return {"counts": {"real": 1000, "fake": 1000}, "clips": clips}

    def _synth_rows(self, splits):
        recipe = st.split_recipe_hash()
        return [{"arm": e["arm"], "generator": e["generator"],
                 "label": e["label"], "path": e["path"],
                 "filename": e["filename"], "sha256": e["sha256"],
                 "recipe_hash": recipe, "score": 0.5, "motion": 0.1,
                 "ttr": 0.5, "lsmi": 1.0, "fps": 24.0,
                 "feature_shape": [16, 196, 768],
                 "unavailable_reason": None}
                for e in splits["clips"]]

    def test_verdict_rejects_row_set_swap(self):
        splits = self._synth_splits()
        good = self._synth_rows(splits)
        self.assertEqual(ex._validate_score_rows(good, splits), [])
        bad = [r for r in good
               if not (r["arm"] == "stage0_real"
                       and r["filename"] == "0000.mp4")]
        bad.append({"arm": "stage0_real", "generator": "genbuster_real",
                    "label": 0, "path": "/tmp/foreign.mp4",
                    "filename": "foreign.mp4", "sha256": "foreign",
                    "recipe_hash": st.split_recipe_hash(), "score": 0.5,
                    "motion": 0.1, "ttr": 0.5, "lsmi": 1.0, "fps": 24.0,
                    "feature_shape": [16, 196, 768],
                    "unavailable_reason": None})
        self.assertEqual(len(bad), len(good))
        problems = ex._validate_score_rows(bad, splits)
        self.assertTrue(any("foreign score keys" in p for p in problems))
        self.assertTrue(any("missing score keys" in p for p in problems))
        with tempfile.TemporaryDirectory() as td:
            with mock.patch.object(ex, "_require_seal", return_value=True), \
                    mock.patch.object(ex, "_load_splits",
                                      return_value=splits), \
                    mock.patch.object(ex, "_load_scores", return_value=bad):
                self.assertNotEqual(ex.verdict(Path(td)), 0)

    def test_seal_refuses_after_scoring(self):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td)
            (out / "scores.jsonl").write_text("{}\n")
            self.assertEqual(ex.write_seal(out), 1)

    @unittest.skipUnless((SEALED_OUT / "seal.txt").is_file(),
                         "sealed artifact absent")
    def test_seal_binds_splits(self):
        with tempfile.TemporaryDirectory() as td:
            dst = Path(td)
            for name in ("seal.txt", "input_hashes.txt", "splits.json",
                         "MANIFEST.json"):
                shutil.copy2(SEALED_OUT / name, dst / name)
            splits = json.loads((dst / "splits.json").read_text())
            splits["clips"][0]["label"] = 1 - int(splits["clips"][0]["label"])
            (dst / "splits.json").write_text(json.dumps(splits, indent=2))
            problems = ex.seal_problems(dst)
            self.assertTrue(any("splits_sha256" in p for p in problems),
                            problems)

    def test_corr_constant_y_returns_none(self):
        rows = [{"fps": 24.0, "score": 1.0}, {"fps": 25.0, "score": 1.0},
                {"fps": 26.0, "score": 1.0}, {"fps": 27.0, "score": 1.0}]
        out = ex._corr(rows)
        self.assertIsNone(out["pearson"])
        self.assertEqual(out["note"], "constant y (score)")

    def test_no_production_reference(self):
        allowed = {
            (WS / "deepfake" / "bench" / "split_t16_temporal.py").resolve(),
            (WS / "deepfake" / "scripts" /
             "exam_split_stage0_t16.py").resolve(),
            (WS / "tests" / "test_split_t16_temporal.py").resolve(),
        }
        roots = [WS / "deepfake", WS / "web"]
        suffixes = {".py", ".js", ".html"}
        needle = "split_t16_temporal"
        offenders = []
        for root in roots:
            if not root.exists():
                continue
            for p in root.rglob("*"):
                if (not p.is_file() or p.suffix.lower() not in suffixes
                        or p.resolve() in allowed
                        or "__pycache__" in p.parts):
                    continue
                try:
                    text = p.read_text(errors="ignore")
                except OSError:
                    continue
                if needle in text:
                    offenders.append(str(p))
        self.assertEqual(offenders, [])


if __name__ == "__main__":
    unittest.main()
