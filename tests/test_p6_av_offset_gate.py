"""P6 AV offset gate kill-test tests (stdlib unittest; no pytest dependency).

Run:
  .venv/bin/python -m unittest discover -s tests -p 'test_p6_av_offset_gate.py' -v
"""

import ast
import csv
import hashlib
import json
import re
import sys
import tempfile
import unittest
from pathlib import Path

WS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WS / "deepfake" / "scripts"))

import p6_av_offset_gate as gate  # noqa: E402 - import after sys.path setup

CAL_PATH = WS / "artifacts" / "restack_video_v5" / "dump_train" / "scores.csv"
EXAM_PATH = (
    WS / "artifacts" / "restack_video_v5" / "gate_avlips" / "features.csv"
)
SCRIPT_PATH = WS / "deepfake" / "scripts" / "p6_av_offset_gate.py"

ALLOWED_IMPORTS = {
    "argparse",
    "csv",
    "datetime",
    "hashlib",
    "json",
    "math",
    "os",
    "pathlib",
    "sys",
    "numpy",
}

CAL_HEADER = [
    "file",
    "sha256_12",
    "label",
    "method",
    "split",
    "avs_syncnet_offset_abs",
    "avs_present",
    "avs_in_coverage",
] + [f"cal_filler_{index}" for index in range(40)]

EXAM_HEADER = [
    "file",
    "sha256_12",
    "label",
    "avs_syncnet_offset_abs",
    "avs_present",
    "avs_in_coverage",
] + [f"exam_filler_{index}" for index in range(41)]

HISTOGRAM_VALUES = (
    [0.5] * 3
    + [1.0] * 16
    + [1.5] * 10
    + [2.0] * 20
    + [2.5] * 3
    + [8.0] * 1
)


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _write_csv(path, header, rows):
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)


def _cal_row(
    offset,
    name,
    label="0",
    method="real",
    split="train",
    present="1",
    coverage="1",
):
    values = {
        "file": f"{name}.mp4",
        "sha256_12": hashlib.sha256(name.encode("utf-8")).hexdigest()[:12],
        "label": label,
        "method": method,
        "split": split,
        "avs_syncnet_offset_abs": f"{offset:.1f}",
        "avs_present": present,
        "avs_in_coverage": coverage,
    }
    return [values.get(column, "0") for column in CAL_HEADER]


def _exam_row(offset, label, name, coverage="1"):
    values = {
        "file": f"{name}.mp4",
        "sha256_12": hashlib.sha256(name.encode("utf-8")).hexdigest()[:12],
        "label": label,
        "avs_syncnet_offset_abs": f"{offset:.1f}",
        "avs_present": "1",
        "avs_in_coverage": coverage,
    }
    return [values.get(column, "0") for column in EXAM_HEADER]


def _build_calibration(path):
    rows = []
    for index, offset in enumerate(HISTOGRAM_VALUES):
        rows.append(_cal_row(offset, f"real{index}"))
    rows.append(_cal_row(2.0, "fake0", label="1", method="fake"))
    rows.append(_cal_row(2.0, "fake1", label="1", method="fake"))
    rows.append(_cal_row(2.0, "absent0", present="0"))
    rows.append(_cal_row(2.0, "uncov0", coverage="0"))
    _write_csv(path, CAL_HEADER, rows)


def _build_exam(path):
    rows = []
    for index in range(358 - 106):
        rows.append(_exam_row(0.0, "0", f"real_low{index}"))
    for index in range(106):
        rows.append(_exam_row(3.0, "0", f"real_high{index}"))
    for index in range(480 - 357):
        rows.append(_exam_row(0.0, "1", f"fake_low{index}"))
    for index in range(357):
        rows.append(_exam_row(3.0, "1", f"fake_high{index}"))
    _write_csv(path, EXAM_HEADER, rows)


def _build_sanity(path):
    rows = []
    for index in range(30):
        coverage = "1" if index < 24 else "0"
        rows.append(_exam_row(0.0, "0", f"s_real{index}", coverage))
    for index in range(30):
        coverage = "1" if index < 23 else "0"
        rows.append(_exam_row(0.0, "1", f"s_fake{index}", coverage))
    _write_csv(path, EXAM_HEADER, rows)


def make_fixtures(root):
    root = Path(root)
    fixtures = {
        "calibration": root / "calibration.csv",
        "exam": root / "exam.csv",
        "parity": root / "parity.json",
        "sanity": root / "sanity.csv",
    }
    _build_calibration(fixtures["calibration"])
    _build_exam(fixtures["exam"])
    _build_sanity(fixtures["sanity"])
    parity = {
        "per_feature_auc": {"avs_syncnet_offset_abs": 0.724},
        "n_reals": 358,
        "n_fakes": 480,
        "n_scored": 838,
    }
    with open(fixtures["parity"], "w", encoding="utf-8") as handle:
        json.dump(parity, handle)
    hashes = {name: _sha256(path) for name, path in fixtures.items()}
    return fixtures, hashes


def gate_argv(fixtures, out):
    return [
        "--calibration",
        str(fixtures["calibration"]),
        "--exam",
        str(fixtures["exam"]),
        "--parity",
        str(fixtures["parity"]),
        "--sanity",
        str(fixtures["sanity"]),
        "--out",
        str(out),
    ]


class GateTestCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.fixtures, self.hashes = make_fixtures(self.root)

    def tearDown(self):
        self._tmp.cleanup()

    def run_fixture(self, out_name="out", hashes=None):
        out = self.root / out_name
        expected = self.hashes if hashes is None else hashes
        code = gate.run_gate(
            gate_argv(self.fixtures, out), expected_hashes=expected
        )
        return code, out

    def read_metrics(self, out):
        return json.loads(
            (out / "metrics.json").read_text(encoding="utf-8")
        )

    def test_tau_quantile_known(self):
        tau, q99 = gate.compute_quantiles(HISTOGRAM_VALUES)
        self.assertEqual(tau, 2.5)
        self.assertLessEqual(abs(q99 - 5.14), 1e-9)

    def test_calibration_filter_semantic(self):
        header, rows = gate.read_csv_rows(self.fixtures["calibration"], 48)
        self.assertEqual(len(header), 48)
        selected = gate.select_calibration(
            rows, str(self.fixtures["calibration"])
        )
        self.assertGreater(len(rows), 53)
        self.assertEqual(len(selected), 53)
        self.assertTrue(all(row["label"] == 0 for row in selected))
        self.assertTrue(all(row["split"] == "train" for row in selected))

    def test_rule_strict_vs_ge(self):
        mask = gate.issued_mask([2.5, 3.0], [1, 1], [1, 1], 2.5)
        self.assertEqual(mask, [False, True])
        header, rows = gate.read_csv_rows(self.fixtures["calibration"], 48)
        selected = gate.select_calibration(
            rows, str(self.fixtures["calibration"])
        )
        ge_count = sum(
            1 for row in selected if row["offset"] >= 2.5
        )
        strict_count = sum(
            1 for row in selected if row["offset"] > 2.5
        )
        self.assertEqual(ge_count, 4)
        self.assertEqual(strict_count, 1)

    def test_auc_average_ranks(self):
        self.assertEqual(
            gate.average_rank_auc([0, 0, 1, 1], [0, 0, 1, 1]), 1.0
        )
        self.assertEqual(
            gate.average_rank_auc([1, 1, 1, 1], [0, 0, 1, 1]), 0.5
        )
        self.assertEqual(
            gate.average_rank_auc([1, 2, 2, 3], [0, 1, 0, 1]), 0.875
        )

    def test_guard_nonempty_out_exit3(self):
        out = self.root / "guarded"
        out.mkdir()
        (out / "stale.txt").write_bytes(b"")
        code = gate.run_gate(
            gate_argv(self.fixtures, out), expected_hashes=self.hashes
        )
        self.assertEqual(code, 3)
        self.assertFalse((out / "metrics.json").exists())

    def test_tau_frozen_before_exam_values(self):
        exam = self.fixtures["exam"]
        with open(exam, "ab") as handle:
            handle.write(b"broken,row\n")
        hashes = dict(self.hashes)
        hashes["exam"] = _sha256(exam)
        code, out = self.run_fixture(hashes=hashes)
        self.assertEqual(code, 3)
        tau_doc = json.loads(
            (out / "tau.json").read_text(encoding="utf-8")
        )
        self.assertEqual(tau_doc["tau_frames"], 2.5)
        self.assertEqual(tau_doc["n"], 53)

    def test_exit_code_kill_vs_invalid(self):
        code, out = self.run_fixture(out_name="kill")
        self.assertEqual(code, 2)
        metrics = self.read_metrics(out)
        self.assertEqual(metrics["verdict"], "KILL")
        self.assertGreater(metrics["exam"]["fp"], 35)
        self.assertLessEqual(
            abs(metrics["exam"]["auc"] - 0.7240), 0.0005
        )
        bad_hashes = dict(self.hashes)
        bad_hashes["exam"] = "0" * 64
        code, out = self.run_fixture(
            out_name="invalid", hashes=bad_hashes
        )
        self.assertEqual(code, 3)
        metrics = self.read_metrics(out)
        self.assertEqual(metrics["verdict"], "INVALID")
        self.assertEqual(metrics["exit_code"], 3)

    def test_metrics_schema_and_atomicity(self):
        code, out = self.run_fixture(out_name="schema")
        self.assertEqual(code, 2)
        document = self.read_metrics(out)
        required = {
            "schema",
            "spec_version",
            "inputs",
            "calibration",
            "exam",
            "sanity",
            "bars",
            "verdict",
            "exit_code",
        }
        self.assertLessEqual(required, set(document))
        self.assertEqual(document["verdict"], "KILL")
        self.assertEqual(document["exit_code"], 2)
        for index in range(1, 10):
            bar = document["bars"][f"B{index}"]
            self.assertLessEqual(
                {"kind", "value", "threshold", "pass"}, set(bar)
            )
            self.assertIn(
                bar["kind"], ("integrity", "substantive", "sentinel")
            )
        for name in ("B7", "B8"):
            bar = document["bars"][name]
            self.assertIsNone(bar["value"])
            self.assertIsNone(bar["pass"])
            self.assertEqual(bar["status"], "PENDING_SENTINEL")
        self.assertEqual(
            sorted(path.name for path in out.iterdir()),
            ["RESULT.md", "metrics.json", "run_meta.json", "tau.json"],
        )
        self.assertFalse(
            any(path.name.endswith(".tmp") for path in out.iterdir())
        )
        result = (out / "RESULT.md").read_text(encoding="utf-8")
        self.assertIn("verdict: PENDING_SENTINEL", result)

    def test_imports_allowlist(self):
        tree = ast.parse(SCRIPT_PATH.read_text(encoding="utf-8"))
        modules = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    modules.add(alias.name.split(".")[0])
            elif isinstance(node, ast.ImportFrom):
                self.assertIsNone(node.level)
                modules.add((node.module or "").split(".")[0])
        self.assertLessEqual(modules, ALLOWED_IMPORTS)

    def test_inputs_unchanged(self):
        before = {
            name: _sha256(path) for name, path in self.fixtures.items()
        }
        code, _ = self.run_fixture(out_name="unchanged")
        self.assertEqual(code, 2)
        after = {
            name: _sha256(path) for name, path in self.fixtures.items()
        }
        self.assertEqual(before, after)

    def test_no_wallclock_in_receipts(self):
        code, out = self.run_fixture(out_name="wallclock")
        self.assertEqual(code, 2)
        pattern = re.compile(r"\d{4}-\d{2}-\d{2}")
        for name in ("metrics.json", "RESULT.md"):
            text = (out / name).read_text(encoding="utf-8")
            self.assertIsNone(pattern.search(text), name)
        document = self.read_metrics(out)
        keys = set()

        def walk(node):
            if isinstance(node, dict):
                for key, value in node.items():
                    keys.add(key.lower())
                    walk(value)
            elif isinstance(node, list):
                for value in node:
                    walk(value)

        walk(document)
        for key in keys:
            self.assertNotIn("time", key)
            self.assertNotIn("date", key)


class RealDataIntegration(unittest.TestCase):
    def test_real_calibration_tau(self):
        header, rows = gate.read_csv_rows(CAL_PATH, 48)
        self.assertEqual(len(header), 48)
        selected = gate.select_calibration(rows, str(CAL_PATH))
        self.assertEqual(len(selected), 53)
        tau, q99 = gate.compute_quantiles(
            [row["offset"] for row in selected]
        )
        self.assertEqual(tau, 2.5)
        self.assertLessEqual(abs(q99 - 5.14), 1e-9)

    def test_real_exam_auc(self):
        labels, offsets, present, coverage = gate.load_exam(EXAM_PATH)
        self.assertEqual(len(labels), 838)
        self.assertEqual(sum(1 for label in labels if label == 0), 358)
        self.assertEqual(sum(1 for label in labels if label == 1), 480)
        auc = gate.average_rank_auc(offsets, labels)
        self.assertLessEqual(abs(auc - 0.7240), 0.0005)


if __name__ == "__main__":
    unittest.main()
