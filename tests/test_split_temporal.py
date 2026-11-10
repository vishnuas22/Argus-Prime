"""SPLIT pre-registration tests (stdlib unittest; no pytest dependency).

Run:
  .venv/bin/python -m unittest discover -s tests -p 'test_split_temporal.py' -v
"""
import hashlib
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

WS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WS / "deepfake" / "bench"))

import split_temporal as st  # noqa: E402

GENBUSTER = WS / "data" / "raw" / "genbuster_plusplus" / "extracted" / "video"
REAL_CLIP = GENBUSTER / "real" / \
    "0025d5c07fb66035a6ae493b2f2ddcf3b9dc7889a47454914bb19645270ea0a7.mp4"


class TestSplitTemporal(unittest.TestCase):
    def test_patch_shape_finite(self):
        feats = st.patch_features(REAL_CLIP)
        self.assertEqual(feats.shape, (8, 196, 768))
        self.assertEqual(feats.dtype, np.float32)
        self.assertTrue(np.isfinite(feats).all())

    def test_frame_indices_formula(self):
        self.assertEqual(st.sample_indices(24.0, 72),
                         [0, 9, 18, 27, 36, 45, 54, 63])
        self.assertEqual(st.sample_indices(3.0, 9),
                         [0, 1, 2, 3, 5, 6, 7, 8])
        self.assertEqual(st.sample_indices(24.0, 60)[-1], 59)

    def test_too_few_frames_fails_closed(self):
        with self.assertRaises(st.SplitUnavailable):
            st.sample_indices(24.0, 7)
        with self.assertRaises(ValueError):
            st.ttr(np.zeros((7, 4, 3), dtype=np.float32))
        with self.assertRaises(ValueError):
            st.lsmi(np.zeros((7, 4, 3), dtype=np.float32))

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

    def test_ttr_formula(self):
        rng = np.random.default_rng(0)
        f = rng.normal(size=(8, 4, 3)).astype(np.float32)
        d1 = f[1:] - f[:-1]
        d2 = f[2:] - f[:-2]
        l1 = np.sum(np.linalg.norm(d1, axis=-1), axis=0)
        l2 = (np.sum(np.linalg.norm(d2, axis=-1), axis=0) / 2.0) \
            * ((8 - 1) / (8 - 2))
        ref = np.log2(l1 + 1e-8) - np.log2(l2 + 1e-8)
        np.testing.assert_allclose(st.ttr(f), ref, rtol=0, atol=1e-6)

    def test_lsmi_formula(self):
        rng = np.random.default_rng(1)
        f = rng.normal(size=(8, 4, 3)).astype(np.float32)
        p = f.reshape(8, 2, 2, 3)
        motion = p[1:] - p[:-1]
        gx = motion[:, :, :-1, :] - motion[:, :, 1:, :]
        gy = motion[:, :-1, :, :] - motion[:, 1:, :, :]
        ref = float((np.linalg.norm(gx, axis=-1).mean()
                     + np.linalg.norm(gy, axis=-1).mean()) / 2.0)
        self.assertAlmostEqual(st.lsmi(f), ref, places=6)

    def test_hash_changes_when_policy_changes(self):
        base = st.split_recipe_hash()
        self.assertEqual(base, st.split_recipe_hash())
        self.assertEqual(len(base), 64)
        variants = [
            {**st.POLICY, "gamma": 7},
            {**st.POLICY, "frames": 16},
            {**st.POLICY, "crop_frac_per_side": 0.05},
            {**st.POLICY, "jpeg": "q80"},
            {**st.POLICY, "eps": 1e-6},
        ]
        hashes = {st.split_recipe_hash(v) for v in variants}
        self.assertEqual(len(hashes), len(variants))
        self.assertNotIn(base, hashes)

    def test_code_sha_is_this_file(self):
        p = Path(st.__file__).resolve()
        self.assertEqual(st.CODE_SHA, hashlib.sha256(p.read_bytes()).hexdigest())

    def test_same_clip_twice_identical(self):
        a = st.score_clip(REAL_CLIP)
        b = st.score_clip(REAL_CLIP)
        for key in ("score", "ttr", "lsmi", "motion", "fps",
                    "n_decoded", "frame_indices"):
            self.assertEqual(a[key], b[key], key)
        self.assertEqual(a["feature_shape"], [8, 196, 768])

    def test_motion_metric(self):
        static = [np.full((64, 64, 3), 128, dtype=np.uint8)
                  for _ in range(8)]
        self.assertEqual(st.motion_metric(static), 0.0)
        rng = np.random.default_rng(2)
        moving = [rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)
                  for _ in range(8)]
        self.assertGreater(st.motion_metric(moving), 0.0)

    def test_preprocess_shape_finite(self):
        rng = np.random.default_rng(3)
        frame = rng.integers(0, 256, (1080, 1920, 3), dtype=np.uint8)
        out = st.preprocess_frame(frame)
        self.assertEqual(out.shape, (3, 224, 224))
        self.assertEqual(out.dtype, np.float32)
        self.assertTrue(np.isfinite(out).all())

    def test_no_production_reference(self):
        allowed = {
            (WS / "deepfake" / "bench" / "split_temporal.py").resolve(),
            (WS / "deepfake" / "scripts" / "exam_split_matched.py").resolve(),
            (WS / "tests" / "test_split_temporal.py").resolve(),
        }
        roots = [WS / "deepfake", WS / "web"]
        suffixes = {".py", ".js", ".html"}
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
                if "split_temporal" in text:
                    offenders.append(str(p))
        self.assertEqual(offenders, [])


if __name__ == "__main__":
    unittest.main()
