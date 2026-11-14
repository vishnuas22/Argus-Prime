#!/usr/bin/env python3
"""P6 AV-sync offset gate kill-test (sealed spec p6-av-offset-gate-v1).

One read-only evaluation of avs_syncnet_offset_abs as a standalone threshold
gate on the AVLips external exam. Writes exactly four receipts under --out.
"""

import argparse
import csv
import datetime
import hashlib
import json
import math
import os
import pathlib
import sys

import numpy as np

WS = pathlib.Path(__file__).resolve().parents[2]
SCHEMA = "p6.av_offset_gate.v1"
SPEC_VERSION = "p6-av-offset-gate-v1"
TAU_SCHEMA = "p6.av_offset_gate.tau.v1"
TAU_FRAMES = 2.5
Q99_FRAMES = 5.14
Q99_TOL = 1e-9
CALIBRATION_N = 53
EXPECTED_HISTOGRAM = {
    "0.5": 3,
    "1.0": 16,
    "1.5": 10,
    "2.0": 20,
    "2.5": 3,
    "8.0": 1,
}
TIES_AT_TAU = 3
SENSITIVITY_GE_N = 4
EXAM_N_ROWS = 838
EXAM_N_REAL = 358
EXAM_N_FAKE = 480
SANITY_N_ROWS = 60
SANITY_N_REAL = 30
SANITY_N_FAKE = 30
AUC_REFERENCE = 0.7240
AUC_PARITY = 0.724
AUC_TOL = 0.0005
FP_MAX = 35
TP_MIN = 144
COVERAGE_MIN = 797
Z95 = 1.959963984540054
CALIBRATION_FILTER = (
    'method == "real" AND avs_present == 1 AND avs_in_coverage == 1'
)
RULE = (
    "issued iff avs_present==1 AND avs_in_coverage==1 "
    "AND offset_abs > 2.5"
)
RECEIPTS = ("tau.json", "metrics.json", "run_meta.json", "RESULT.md")

_RESTACK = WS / "artifacts" / "restack_video_v5"
DEFAULT_CALIBRATION = _RESTACK / "dump_train" / "scores.csv"
DEFAULT_EXAM = _RESTACK / "gate_avlips" / "features.csv"
DEFAULT_PARITY = _RESTACK / "gate_avlips" / "metrics.json"
DEFAULT_SANITY = _RESTACK / "gate_av" / "features.csv"
DEFAULT_OUT = WS / "artifacts" / "p6_av_gate" / "run"

EXPECTED_SHA256 = {
    "calibration": (
        "da53b6fdacecd4141a53b8df82e35466cdeb1e856a00942fb9a824913eae9d1d"
    ),
    "exam": (
        "60e3471fe824cc979c1b5b19636730f601d97f855c329aa6968874781f95637f"
    ),
    "parity": (
        "366c7bcb4678be357acf9574433c1b7ad0851d28997fb472833ea44d977fc812"
    ),
    "sanity": (
        "6d820c3a459716bd6ad65578c55ed2b4be8d72778569564e2bb8150928140364"
    ),
}

CALIBRATION_COLUMNS = (
    "file",
    "sha256_12",
    "label",
    "method",
    "split",
    "avs_syncnet_offset_abs",
    "avs_present",
    "avs_in_coverage",
)
EXAM_COLUMNS = (
    "sha256_12",
    "label",
    "avs_syncnet_offset_abs",
    "avs_present",
    "avs_in_coverage",
)
REQUIRED_SECTIONS = (
    "## Pre-registered rule",
    "## Inputs",
    "## Calibration",
    "## Exam",
    "## Bars",
    "## Sensitivity (report-only)",
    "## Reproduction",
    "## Sentinel checklist",
)


class GateError(Exception):
    """Sealed-contract violation."""


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv_rows(path, expected_cols):
    path = pathlib.Path(path)
    with open(path, "rb") as handle:
        head = handle.read(3)
    if head == b"\xef\xbb\xbf":
        raise GateError(f"BOM not allowed: {path}")
    try:
        with open(path, "r", encoding="utf-8", newline="") as handle:
            reader = csv.reader(handle)
            try:
                header = [field.strip() for field in next(reader)]
            except StopIteration as exc:
                raise GateError(f"empty CSV: {path}") from exc
            if len(header) != expected_cols:
                raise GateError(
                    f"{path}: header has {len(header)} columns, "
                    f"expected {expected_cols}"
                )
            if len(set(header)) != len(header):
                raise GateError(f"{path}: duplicate column names")
            rows = []
            for line_no, raw_row in enumerate(reader, start=2):
                if not raw_row:
                    raise GateError(f"{path}:{line_no}: blank line")
                if len(raw_row) != len(header):
                    raise GateError(
                        f"{path}:{line_no}: {len(raw_row)} fields, "
                        f"expected {len(header)}"
                    )
                rows.append(
                    {
                        name: value.strip()
                        for name, value in zip(header, raw_row)
                    }
                )
    except UnicodeDecodeError as exc:
        raise GateError(f"not valid UTF-8: {path}: {exc}") from exc
    return header, rows


def require_columns(header, required, path):
    missing = [name for name in required if name not in header]
    if missing:
        raise GateError(f"{path}: missing columns {missing}")


def parse_int01(value, where):
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise GateError(f"{where}: not an integer: {value!r}") from exc
    if number not in (0, 1):
        raise GateError(f"{where}: value outside {{0,1}}: {number}")
    return number


def parse_offset(value, where):
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise GateError(f"{where}: not a float: {value!r}") from exc
    if not math.isfinite(number):
        raise GateError(f"{where}: non-finite offset: {number}")
    if abs(2.0 * number - round(2.0 * number)) > 1e-9:
        raise GateError(f"{where}: offset not a multiple of 0.5: {number}")
    return number


def select_calibration(rows, source):
    selected = []
    for index, row in enumerate(rows):
        where = f"{source}:{index + 2}"
        present = parse_int01(row["avs_present"], where + " avs_present")
        coverage = parse_int01(
            row["avs_in_coverage"], where + " avs_in_coverage"
        )
        label = parse_int01(row["label"], where + " label")
        offset = parse_offset(row["avs_syncnet_offset_abs"], where)
        if row["method"] == "real" and present == 1 and coverage == 1:
            selected.append(
                {"label": label, "split": row["split"], "offset": offset}
            )
    return selected


def offset_histogram(values):
    counts = {}
    for value in values:
        key = float(value)
        counts[key] = counts.get(key, 0) + 1
    return {f"{key:.1f}": counts[key] for key in sorted(counts)}


def compute_quantiles(values):
    array = np.asarray(values, dtype=np.float64)
    tau = float(np.quantile(array, 0.95, method="linear"))
    q99 = float(np.quantile(array, 0.99, method="linear"))
    return tau, q99


def average_rank_auc(scores, labels):
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    count = len(scores)
    if count != len(labels):
        raise GateError("AUC: scores and labels differ in length")
    n1 = int(labels.sum())
    n0 = count - n1
    if n0 == 0 or n1 == 0:
        raise GateError("AUC: both classes are required")
    order = np.argsort(scores, kind="mergesort")
    ordered = scores[order]
    ranks = np.empty(count, dtype=np.float64)
    start = 0
    while start < count:
        stop = start
        while stop + 1 < count and ordered[stop + 1] == ordered[start]:
            stop += 1
        ranks[start : stop + 1] = 0.5 * (start + stop) + 1.0
        start = stop + 1
    restored = np.empty(count, dtype=np.float64)
    restored[order] = ranks
    rank_sum = float(restored[labels == 1].sum())
    return (rank_sum - n1 * (n1 + 1) / 2.0) / (n1 * n0)


def issued_mask(offsets, present, coverage, tau):
    mask = []
    for offset, is_present, is_covered in zip(offsets, present, coverage):
        mask.append(
            is_present == 1 and is_covered == 1 and offset > tau
        )
    return mask


def ge_mask(offsets, present, coverage, tau):
    mask = []
    for offset, is_present, is_covered in zip(offsets, present, coverage):
        mask.append(
            is_present == 1 and is_covered == 1 and offset >= tau
        )
    return mask


def wilson_interval(successes, total):
    if total <= 0:
        raise GateError("Wilson interval needs a positive denominator")
    p = successes / total
    z = Z95
    denominator = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / denominator
    half = (
        z
        * math.sqrt(
            p * (1.0 - p) / total + z * z / (4.0 * total * total)
        )
        / denominator
    )
    return [max(0.0, center - half), min(1.0, center + half)]


def load_exam(path):
    header, rows = read_csv_rows(path, 47)
    require_columns(header, EXAM_COLUMNS, path)
    labels, offsets, present, coverage = [], [], [], []
    for index, row in enumerate(rows):
        where = f"{path}:{index + 2}"
        labels.append(parse_int01(row["label"], where + " label"))
        present.append(
            parse_int01(row["avs_present"], where + " avs_present")
        )
        coverage.append(
            parse_int01(row["avs_in_coverage"], where + " avs_in_coverage")
        )
        offsets.append(parse_offset(row["avs_syncnet_offset_abs"], where))
    return labels, offsets, present, coverage


def load_sanity(path):
    header, rows = read_csv_rows(path, 47)
    require_columns(header, EXAM_COLUMNS, path)
    labels, present, coverage = [], [], []
    for index, row in enumerate(rows):
        where = f"{path}:{index + 2}"
        labels.append(parse_int01(row["label"], where + " label"))
        present.append(
            parse_int01(row["avs_present"], where + " avs_present")
        )
        coverage.append(
            parse_int01(row["avs_in_coverage"], where + " avs_in_coverage")
        )
        parse_offset(row["avs_syncnet_offset_abs"], where)
    return labels, present, coverage


def load_parity(path):
    with open(path, "rb") as handle:
        raw = handle.read()
    if raw.startswith(b"\xef\xbb\xbf"):
        raise GateError(f"BOM not allowed: {path}")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise GateError(f"not valid UTF-8: {path}: {exc}") from exc
    try:
        document = json.loads(text)
    except json.JSONDecodeError as exc:
        raise GateError(f"{path}: invalid JSON: {exc}") from exc
    try:
        auc = document["per_feature_auc"]["avs_syncnet_offset_abs"]
        n_reals = document["n_reals"]
        n_fakes = document["n_fakes"]
        n_scored = document["n_scored"]
    except (KeyError, TypeError) as exc:
        raise GateError(f"{path}: missing parity key: {exc}") from exc
    if auc != AUC_PARITY:
        raise GateError(f"{path}: parity auc {auc!r} != {AUC_PARITY}")
    if (n_reals, n_fakes, n_scored) != (
        EXAM_N_REAL,
        EXAM_N_FAKE,
        EXAM_N_ROWS,
    ):
        raise GateError(
            f"{path}: parity counts {(n_reals, n_fakes, n_scored)} "
            f"!= {(EXAM_N_REAL, EXAM_N_FAKE, EXAM_N_ROWS)}"
        )
    return {
        "auc": auc,
        "n_reals": n_reals,
        "n_fakes": n_fakes,
        "n_scored": n_scored,
        "n_lines": len(text.splitlines()),
    }


def dump_json(document):
    return json.dumps(document, indent=2, sort_keys=True) + "\n"


def write_atomic(target, text):
    target = pathlib.Path(target)
    tmp = target.with_suffix(target.suffix + ".tmp")
    try:
        with open(tmp, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, target)
    except BaseException:
        try:
            if tmp.exists():
                tmp.unlink()
        except OSError:
            pass
        raise


def _is_within(child, parent):
    try:
        child.relative_to(parent)
    except ValueError:
        return False
    return True


def check_out_dir(out, inputs, force):
    out_resolved = out.resolve()
    if out_resolved == WS.resolve():
        print(
            f"ERROR: out dir must not be the workspace root: {out}",
            file=sys.stderr,
        )
        return 3
    for name, path in inputs.items():
        input_resolved = path.resolve()
        if _is_within(input_resolved, out_resolved) or _is_within(
            out_resolved, input_resolved
        ):
            print(
                f"ERROR: out dir overlaps input {name}: {out}",
                file=sys.stderr,
            )
            return 3
    if out.exists() and any(out.iterdir()) and not force:
        print(
            f"ERROR: out dir is not empty: {out} (pass --force)",
            file=sys.stderr,
        )
        return 3
    return None


def make_bar(kind, value, threshold, passed):
    return {
        "kind": kind,
        "value": value,
        "threshold": threshold,
        "pass": passed,
    }


def resolve_verdict(bars):
    integrity_failed = any(
        bars[name]["pass"] is False for name in ("B1", "B2", "B6", "B9")
    )
    substantive_failed = any(
        bars[name]["pass"] is False for name in ("B3", "B4", "B5")
    )
    if integrity_failed:
        return "INVALID", 3
    if substantive_failed:
        return "KILL", 2
    return "PASS", 0


def validate_metrics(document):
    required = (
        "schema",
        "spec_version",
        "inputs",
        "calibration",
        "exam",
        "sanity",
        "bars",
        "verdict",
        "exit_code",
    )
    missing = [key for key in required if key not in document]
    if missing:
        raise GateError(f"metrics: missing keys {missing}")
    if document["schema"] != SCHEMA:
        raise GateError("metrics: schema mismatch")
    if document["spec_version"] != SPEC_VERSION:
        raise GateError("metrics: spec_version mismatch")
    if set(document["inputs"]) != {
        "calibration",
        "exam",
        "parity",
        "sanity",
    }:
        raise GateError("metrics: inputs roles mismatch")
    for role, entry in document["inputs"].items():
        for key in ("path", "sha256", "rows"):
            if key not in entry:
                raise GateError(f"metrics: inputs.{role} missing {key}")
    calibration_keys = (
        "filter",
        "n",
        "histogram",
        "tau_frames",
        "q99_frames",
        "rule",
        "ties_at_tau",
        "sensitivity_ge",
    )
    for key in calibration_keys:
        if key not in document["calibration"]:
            raise GateError(f"metrics: calibration missing {key}")
    exam_keys = (
        "n_rows",
        "n_real",
        "n_fake",
        "n_in_coverage",
        "coverage",
        "auc",
        "auc_reference",
        "auc_delta",
        "fp",
        "fp_den",
        "fp_rate",
        "fp_wilson95",
        "tp",
        "tp_den",
        "tp_rate",
        "tp_wilson95",
        "ties_at_tau_real",
        "ties_at_tau_fake",
        "sensitivity_ge",
    )
    for key in exam_keys:
        if key not in document["exam"]:
            raise GateError(f"metrics: exam missing {key}")
    for key in ("fp", "tp", "fp_rate", "tp_rate"):
        if key not in document["exam"]["sensitivity_ge"]:
            raise GateError(f"metrics: exam sensitivity_ge missing {key}")
    sanity_keys = ("n_rows", "n_real", "n_fake", "n_in_coverage")
    for key in sanity_keys:
        if key not in document["sanity"]:
            raise GateError(f"metrics: sanity missing {key}")
    if document["sanity"].get("report_only") is not True:
        raise GateError("metrics: sanity report_only must be true")
    bars = document["bars"]
    for index in range(1, 10):
        name = f"B{index}"
        bar = bars.get(name)
        if not isinstance(bar, dict):
            raise GateError(f"metrics: missing bar {name}")
        for key in ("kind", "value", "threshold", "pass"):
            if key not in bar:
                raise GateError(f"metrics: bar {name} missing {key}")
        if bar["kind"] not in ("integrity", "substantive", "sentinel"):
            raise GateError(f"metrics: bar {name} has bad kind")
    for name in ("B1", "B2", "B3", "B4", "B5", "B6", "B9"):
        if not isinstance(bars[name]["pass"], bool):
            raise GateError(f"metrics: bar {name} pass must be bool")
    for name in ("B7", "B8"):
        bar = bars[name]
        if bar["value"] is not None or bar["pass"] is not None:
            raise GateError(f"metrics: bar {name} must be pending")
        if bar.get("status") != "PENDING_SENTINEL":
            raise GateError(f"metrics: bar {name} status mismatch")
    expected_exit = {"PASS": 0, "KILL": 2, "INVALID": 3}
    verdict = document["verdict"]
    if verdict not in expected_exit:
        raise GateError(f"metrics: bad verdict {verdict!r}")
    if document["exit_code"] != expected_exit[verdict]:
        raise GateError("metrics: exit_code disagrees with verdict")


def validate_result_text(text):
    lines = text.splitlines()
    if not lines:
        raise GateError("RESULT.md: empty")
    if lines[0] != "# P6 AV offset gate — kill-test result":
        raise GateError("RESULT.md: title mismatch")
    if lines[1] != "verdict: PENDING_SENTINEL":
        raise GateError("RESULT.md: literal verdict line missing")
    positions = []
    for section in REQUIRED_SECTIONS:
        if section not in lines:
            raise GateError(f"RESULT.md: missing section {section}")
        positions.append(lines.index(section))
    if positions != sorted(positions):
        raise GateError("RESULT.md: sections out of order")


def _pct(value):
    return f"{value * 100.0:.4f}%"


def _render_bar_pass(value):
    if value is None:
        return "PENDING"
    return "PASS" if value else "FAIL"


def render_result(metrics):
    calibration = metrics["calibration"]
    exam = metrics["exam"]
    inputs = metrics["inputs"]
    sensitivity_ge = calibration["sensitivity_ge"]
    exam_ge = exam["sensitivity_ge"]
    lines = [
        "# P6 AV offset gate — kill-test result",
        "verdict: PENDING_SENTINEL",
        f"script_verdict: {metrics['verdict']}",
        f"script_exit: {metrics['exit_code']}",
        "",
        "## Pre-registered rule",
        "",
        f"`{calibration['rule']}`",
        "",
        f"- tau = {calibration['tau_frames']} frames (q95, linear)",
        f"- q99 = {calibration['q99_frames']} frames",
        "- direction: higher offset = fake-ward (no inversion)",
        "- ties at tau are not issued (strict `>`)",
        "",
        "## Inputs",
        "",
        "| role | path | sha256 | rows |",
        "|---|---|---|---|",
    ]
    for role in ("calibration", "exam", "parity", "sanity"):
        entry = inputs[role]
        lines.append(
            f"| {role} | {entry['path']} | {entry['sha256']} "
            f"| {entry['rows']} |"
        )
    lines.extend(
        [
            "",
            "## Calibration",
            "",
            f"- filter: {calibration['filter']}",
            f"- n: {calibration['n']}",
            "- histogram: "
            + json.dumps(calibration["histogram"], sort_keys=True),
            f"- tau_frames: {calibration['tau_frames']}",
            f"- q99_frames: {calibration['q99_frames']}",
            f"- ties_at_tau: {calibration['ties_at_tau']}",
            "- sensitivity_ge: "
            f"{sensitivity_ge['n_issue']}/{calibration['n']} = "
            f"{_pct(sensitivity_ge['frac'])}",
            "",
            "## Exam",
            "",
            f"- n_rows: {exam['n_rows']} "
            f"(real {exam['n_real']} / fake {exam['n_fake']})",
            f"- n_in_coverage: {exam['n_in_coverage']} "
            f"({exam['coverage']:.6f})",
            f"- auc: {exam['auc']:.6f} "
            f"(reference {exam['auc_reference']}, "
            f"delta {exam['auc_delta']:+.6f})",
            f"- fp: {exam['fp']}/{exam['fp_den']} = {_pct(exam['fp_rate'])} "
            f"(Wilson 95% CI {exam['fp_wilson95'][0]:.6f}-"
            f"{exam['fp_wilson95'][1]:.6f})",
            f"- tp: {exam['tp']}/{exam['tp_den']} = {_pct(exam['tp_rate'])} "
            f"(Wilson 95% CI {exam['tp_wilson95'][0]:.6f}-"
            f"{exam['tp_wilson95'][1]:.6f})",
            f"- ties at tau: real {exam['ties_at_tau_real']} / "
            f"fake {exam['ties_at_tau_fake']}",
            "",
            "## Bars",
            "",
            "| bar | kind | value | threshold | pass |",
            "|---|---|---|---|---|",
        ]
    )
    for index in range(1, 10):
        name = f"B{index}"
        bar = metrics["bars"][name]
        value = (
            "PENDING" if bar["value"] is None else json.dumps(bar["value"])
        )
        lines.append(
            f"| {name} | {bar['kind']} | {value} | {bar['threshold']} "
            f"| {_render_bar_pass(bar['pass'])} |"
        )
    lines.extend(
        [
            "",
            "## Sensitivity (report-only)",
            "",
            f"- calibration >= tau: {sensitivity_ge['n_issue']}/"
            f"{calibration['n']} = {_pct(sensitivity_ge['frac'])}",
            f"- exam >= tau: fp {exam_ge['fp']}/{exam['fp_den']} = "
            f"{_pct(exam_ge['fp_rate'])}; tp {exam_ge['tp']}/"
            f"{exam['tp_den']} = {_pct(exam_ge['tp_rate'])}",
            "Rule switch after exam results: FORBIDDEN.",
            "",
            "## Reproduction",
            "",
            "```",
            "./.venv/bin/python deepfake/scripts/p6_av_offset_gate.py "
            "--out artifacts/p6_av_gate/run",
            "```",
            "",
            "Input sha256:",
        ]
    )
    for role in ("calibration", "exam", "parity", "sanity"):
        lines.append(f"- {role}: {inputs[role]['sha256']}")
    lines.extend(
        [
            "",
            "## Sentinel checklist",
            "",
            "- [ ] 1. Recompute four sha256 values; match §3.",
            "- [ ] 2. Re-extract the 53 calibration rows; confirm histogram.",
            "- [ ] 3. Confirm tau 2.5 and q99 5.14 at 1e-9.",
            "- [ ] 4. Recompute AUC independently; match B2.",
            "- [ ] 5. Recount FP/TP; confirm denominators 358/480.",
            "- [ ] 6. Confirm B5 uses all 838 rows.",
            "- [ ] 7. Two fresh runs; cmp metrics.json and RESULT.md.",
            "- [ ] 8. Re-hash inputs after both runs; unchanged.",
            "- [ ] 9. unittest 0 failures/errors/skips; ruff 0 findings.",
            "- [ ] 10. Baseline-relative git state exact.",
            "- [ ] 11. Sections and literal verdict line; no memory edits.",
            "- [ ] 12. Four files, no .tmp, no writes outside the out dir.",
            "",
        ]
    )
    return "\n".join(lines)


def execute(out, inputs, expected, argv):
    out.mkdir(parents=True, exist_ok=True)
    input_sha = {name: sha256_file(path) for name, path in inputs.items()}
    for name in inputs:
        if input_sha[name] != expected[name]:
            raise GateError(
                f"{name} sha256 mismatch: {input_sha[name]} "
                f"!= {expected[name]}"
            )

    calibration_path = inputs["calibration"]
    header, rows = read_csv_rows(calibration_path, 48)
    require_columns(header, CALIBRATION_COLUMNS, calibration_path)
    selected = select_calibration(rows, str(calibration_path))
    if len(selected) != CALIBRATION_N:
        raise GateError(f"calibration n={len(selected)}, expected 53")
    for index, row in enumerate(selected):
        if row["label"] != 0:
            raise GateError(f"calibration row {index}: label != 0")
        if row["split"] != "train":
            raise GateError(f"calibration row {index}: split != train")
    values = [row["offset"] for row in selected]
    histogram = offset_histogram(values)
    if histogram != EXPECTED_HISTOGRAM:
        raise GateError(f"calibration histogram mismatch: {histogram}")
    tau, q99 = compute_quantiles(values)
    if tau != TAU_FRAMES:
        raise GateError(f"tau={tau!r}, expected 2.5")
    if abs(q99 - Q99_FRAMES) > Q99_TOL:
        raise GateError(f"q99={q99!r}, expected 5.14")
    ties = sum(1 for value in values if value == tau)
    if ties != TIES_AT_TAU:
        raise GateError(f"ties_at_tau={ties}, expected 3")
    ge_count = sum(1 for value in values if value >= tau)
    if ge_count != SENSITIVITY_GE_N:
        raise GateError(f"sensitivity_ge={ge_count}, expected 4")

    tau_document = {
        "schema": TAU_SCHEMA,
        "tau_frames": tau,
        "q99_frames": round(q99, 12),
        "n": len(selected),
        "histogram": histogram,
        "quantile_method": "linear",
        "ties_at_tau": ties,
        "sensitivity_ge": {
            "n_issue": ge_count,
            "frac": ge_count / len(selected),
        },
        "rule": RULE,
    }
    write_atomic(out / "tau.json", dump_json(tau_document))

    parity = load_parity(inputs["parity"])
    sanity_labels, sanity_present, sanity_coverage = load_sanity(
        inputs["sanity"]
    )
    sanity_n_rows = len(sanity_labels)
    sanity_n_real = sum(1 for label in sanity_labels if label == 0)
    sanity_n_fake = sanity_n_rows - sanity_n_real
    if (sanity_n_rows, sanity_n_real, sanity_n_fake) != (
        SANITY_N_ROWS,
        SANITY_N_REAL,
        SANITY_N_FAKE,
    ):
        raise GateError(
            "sanity counts "
            f"{(sanity_n_rows, sanity_n_real, sanity_n_fake)}"
        )
    sanity_n_in_coverage = sum(
        1
        for is_present, is_covered in zip(sanity_present, sanity_coverage)
        if is_present == 1 and is_covered == 1
    )

    labels, offsets, present, coverage = load_exam(inputs["exam"])
    n_rows = len(labels)
    n_real = sum(1 for label in labels if label == 0)
    n_fake = n_rows - n_real
    if (n_rows, n_real, n_fake) != (
        EXAM_N_ROWS,
        EXAM_N_REAL,
        EXAM_N_FAKE,
    ):
        raise GateError(f"exam counts {(n_rows, n_real, n_fake)}")
    n_in_coverage = sum(
        1
        for is_present, is_covered in zip(present, coverage)
        if is_present == 1 and is_covered == 1
    )
    auc = average_rank_auc(offsets, labels)
    mask = issued_mask(offsets, present, coverage, TAU_FRAMES)
    fp = sum(1 for flag, label in zip(mask, labels) if flag and label == 0)
    tp = sum(1 for flag, label in zip(mask, labels) if flag and label == 1)
    ties_real = sum(
        1
        for offset, label in zip(offsets, labels)
        if label == 0 and offset == TAU_FRAMES
    )
    ties_fake = sum(
        1
        for offset, label in zip(offsets, labels)
        if label == 1 and offset == TAU_FRAMES
    )
    ge_flags = ge_mask(offsets, present, coverage, TAU_FRAMES)
    fp_ge = sum(
        1 for flag, label in zip(ge_flags, labels) if flag and label == 0
    )
    tp_ge = sum(
        1 for flag, label in zip(ge_flags, labels) if flag and label == 1
    )

    after_sha = {name: sha256_file(path) for name, path in inputs.items()}
    b6_pass = after_sha == input_sha
    no_tmp = not any(path.name.endswith(".tmp") for path in out.iterdir())

    bars = {
        "B1": make_bar("integrity", True, "all exact", True),
        "B2": make_bar(
            "integrity",
            round(auc, 6),
            "abs(auc - 0.7240) <= 0.0005",
            abs(auc - AUC_REFERENCE) <= AUC_TOL,
        ),
        "B3": make_bar(
            "substantive", fp, "FP <= 35 (den 358)", fp <= FP_MAX
        ),
        "B4": make_bar(
            "substantive", tp, "TP >= 144 (den 480)", tp >= TP_MIN
        ),
        "B5": make_bar(
            "substantive",
            n_in_coverage,
            "n_in_coverage >= 797 (den 838)",
            n_in_coverage >= COVERAGE_MIN,
        ),
        "B6": make_bar(
            "integrity",
            b6_pass,
            "B6a inputs unchanged; B6b byte-identical runs (sentinel)",
            b6_pass,
        ),
        "B7": {
            "kind": "sentinel",
            "value": None,
            "threshold": "0 failures / 0 errors / 0 skips; ruff 0 findings",
            "pass": None,
            "status": "PENDING_SENTINEL",
        },
        "B8": {
            "kind": "sentinel",
            "value": None,
            "threshold": "baseline-relative git state exact",
            "pass": None,
            "status": "PENDING_SENTINEL",
        },
        "B9": make_bar(
            "integrity",
            no_tmp,
            "required keys/sections; no .tmp",
            no_tmp,
        ),
    }
    verdict, exit_code = resolve_verdict(bars)

    metrics = {
        "schema": SCHEMA,
        "spec_version": SPEC_VERSION,
        "inputs": {
            "calibration": {
                "path": str(calibration_path),
                "sha256": input_sha["calibration"],
                "rows": len(selected),
            },
            "exam": {
                "path": str(inputs["exam"]),
                "sha256": input_sha["exam"],
                "rows": n_rows,
            },
            "parity": {
                "path": str(inputs["parity"]),
                "sha256": input_sha["parity"],
                "rows": parity["n_lines"],
            },
            "sanity": {
                "path": str(inputs["sanity"]),
                "sha256": input_sha["sanity"],
                "rows": sanity_n_rows,
            },
        },
        "calibration": {
            "filter": CALIBRATION_FILTER,
            "n": len(selected),
            "histogram": histogram,
            "tau_frames": tau,
            "q99_frames": round(q99, 12),
            "rule": RULE,
            "ties_at_tau": ties,
            "sensitivity_ge": {
                "n_issue": ge_count,
                "frac": ge_count / len(selected),
            },
        },
        "exam": {
            "n_rows": n_rows,
            "n_real": n_real,
            "n_fake": n_fake,
            "n_in_coverage": n_in_coverage,
            "coverage": n_in_coverage / EXAM_N_ROWS,
            "auc": round(auc, 6),
            "auc_reference": AUC_PARITY,
            "auc_delta": round(auc - AUC_REFERENCE, 6),
            "fp": fp,
            "fp_den": EXAM_N_REAL,
            "fp_rate": fp / EXAM_N_REAL,
            "fp_wilson95": wilson_interval(fp, EXAM_N_REAL),
            "tp": tp,
            "tp_den": EXAM_N_FAKE,
            "tp_rate": tp / EXAM_N_FAKE,
            "tp_wilson95": wilson_interval(tp, EXAM_N_FAKE),
            "ties_at_tau_real": ties_real,
            "ties_at_tau_fake": ties_fake,
            "sensitivity_ge": {
                "fp": fp_ge,
                "tp": tp_ge,
                "fp_rate": fp_ge / EXAM_N_REAL,
                "tp_rate": tp_ge / EXAM_N_FAKE,
            },
        },
        "sanity": {
            "n_rows": sanity_n_rows,
            "n_real": sanity_n_real,
            "n_fake": sanity_n_fake,
            "n_in_coverage": sanity_n_in_coverage,
            "report_only": True,
        },
        "bars": bars,
        "verdict": verdict,
        "exit_code": exit_code,
    }
    validate_metrics(metrics)
    result_text = render_result(metrics)
    validate_result_text(result_text)

    write_atomic(out / "metrics.json", dump_json(metrics))
    extras = sorted(
        path.name for path in out.iterdir() if path.name not in RECEIPTS
    )
    run_meta = {
        "generated_utc": datetime.datetime.now(
            datetime.timezone.utc
        ).isoformat(timespec="seconds"),
        "argv": [str(pathlib.Path(__file__).resolve())] + list(argv),
        "cwd": os.getcwd(),
        "python_version": sys.version.split()[0],
        "numpy_version": np.__version__,
        "platform": sys.platform,
        "script_sha256": sha256_file(pathlib.Path(__file__).resolve()),
        "out_extra_files": extras,
    }
    write_atomic(out / "run_meta.json", dump_json(run_meta))
    write_atomic(out / "RESULT.md", result_text)
    leftovers = sorted(
        path.name for path in out.iterdir() if path.name.endswith(".tmp")
    )
    if leftovers:
        raise GateError(f".tmp leftovers: {leftovers}")
    print(
        json.dumps(
            {
                "schema": SCHEMA,
                "verdict": metrics["verdict"],
                "exit_code": metrics["exit_code"],
            },
            sort_keys=True,
        )
    )
    return metrics["exit_code"]


class _Parser(argparse.ArgumentParser):
    def error(self, message):
        self.print_usage(sys.stderr)
        raise SystemExit(3)


def build_parser():
    parser = _Parser(
        description="P6 AV-sync offset gate kill-test (read-only)."
    )
    parser.add_argument(
        "--calibration",
        default=str(DEFAULT_CALIBRATION),
        help="calibration scores.csv",
    )
    parser.add_argument(
        "--exam",
        default=str(DEFAULT_EXAM),
        help="AVLips exam features.csv",
    )
    parser.add_argument(
        "--parity",
        default=str(DEFAULT_PARITY),
        help="AVLips parity metrics.json",
    )
    parser.add_argument(
        "--sanity",
        default=str(DEFAULT_SANITY),
        help="sanity features.csv (report-only)",
    )
    parser.add_argument(
        "--out",
        default=str(DEFAULT_OUT),
        help="receipt directory",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="overwrite a non-empty out dir",
    )
    return parser


def write_invalid_receipt(out, exc):
    message = f"{type(exc).__name__}: {exc}"
    try:
        out.mkdir(parents=True, exist_ok=True)
        receipt = {
            "schema": SCHEMA,
            "spec_version": SPEC_VERSION,
            "verdict": "INVALID",
            "exit_code": 3,
            "error": message,
        }
        write_atomic(out / "metrics.json", dump_json(receipt))
    except Exception:
        pass
    print(f"ERROR: {message}", file=sys.stderr)
    return 3


def run_gate(argv=None, expected_hashes=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    args = build_parser().parse_args(argv)
    out = pathlib.Path(args.out)
    inputs = {
        "calibration": pathlib.Path(args.calibration),
        "exam": pathlib.Path(args.exam),
        "parity": pathlib.Path(args.parity),
        "sanity": pathlib.Path(args.sanity),
    }
    expected = dict(
        EXPECTED_SHA256 if expected_hashes is None else expected_hashes
    )
    guard_code = check_out_dir(out, inputs, args.force)
    if guard_code is not None:
        return guard_code
    try:
        return execute(out, inputs, expected, argv)
    except Exception as exc:
        return write_invalid_receipt(out, exc)


def main(argv=None):
    return run_gate(argv)


if __name__ == "__main__":
    sys.exit(main())
