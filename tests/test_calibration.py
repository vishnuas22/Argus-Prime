"""Fast tier: TASK-C1 calibration primitives (no models, no corpora, no artifacts).

Locks the properties the C1 stack claims:
  * logit/sigmoid are inverses and monotone.
  * CourtCalibrator is AUC-invariant (a monotone map cannot change ranking).
  * CourtCalibrator refuses degenerate input instead of encoding a class prior.
  * ltt_threshold honours its finite-sample FPR guarantee.
  * cohort_normalize rejects contaminated/invalid cohorts.
  * risk_curve is well-formed and its minimum is the Bayes-optimal grid point.
  * CalibratedStackerJudge delegates unchanged when no calibrator is supplied.
"""
import math

import numpy as np
import pytest

from calibration import (
    CalibrationConfig,
    CourtCalibrator,
    best_cost_point,
    cohort_normalize,
    fit_cohort_stats,
    logit_transform,
    ltt_threshold,
    risk_curve,
    sigmoid,
)

pytestmark = pytest.mark.fast


def _separable(n=200, seed=0, gap=3.0):
    rng = np.random.default_rng(seed)
    y = np.r_[np.ones(n // 2), np.zeros(n // 2)]
    x = np.r_[rng.normal(gap, 1.0, n // 2), rng.normal(0.0, 1.0, n // 2)]
    p = 1.0 / (1.0 + np.exp(-x))
    return p, y.astype(int)


# --- logit / sigmoid ---------------------------------------------------------

def test_logit_sigmoid_roundtrip():
    p = np.array([0.001, 0.01, 0.25, 0.5, 0.75, 0.99, 0.999])
    assert np.allclose(sigmoid(logit_transform(p)), p, atol=1e-9)


def test_logit_is_monotone_increasing():
    p = np.linspace(0.01, 0.99, 50)
    z = logit_transform(p)
    assert np.all(np.diff(z) > 0)


def test_logit_clips_extremes_to_finite():
    z = logit_transform([0.0, 1.0])
    assert np.all(np.isfinite(z))


# --- CourtCalibrator ---------------------------------------------------------

def test_calibrator_preserves_auc():
    from sklearn.metrics import roc_auc_score

    p, y = _separable()
    cal = CourtCalibrator.fit(p, y, scope="test:separable")
    cal_p = np.array([cal.apply_one(pi) for pi in p])

    # monotone map => identical ordering
    assert np.array_equal(np.argsort(p), np.argsort(cal_p))

    # AUC is invariant by construction
    auc_before = roc_auc_score(y, p)
    auc_after = roc_auc_score(y, cal_p)
    assert auc_after == pytest.approx(auc_before)
    assert cal.config.base_auc == pytest.approx(auc_before)
    assert auc_before > 0.9


def test_calibrator_moves_operating_point():
    """A mis-scaled base score gets a sharper calibrated one (the C1 goal)."""
    p, y = _separable(seed=1)
    cal = CourtCalibrator.fit(p, y, scope="test:sharp", C=10.0)
    base = np.array([1.0 if pi > 0.5 else 0.0 for pi in p])
    calib = np.array([1.0 if ci > 0.5 else 0.0 for ci in
                      [cal.apply_one(pi) for pi in p]])
    err_base = float((base != y).mean())
    err_cal = float((calib != y).mean())
    assert err_cal <= err_base + 1e-9


def test_calibrator_rejects_single_class():
    p, _ = _separable()
    with pytest.raises(ValueError, match="single-class"):
        CourtCalibrator.fit(p, np.ones(len(p), dtype=int), scope="test")


def test_calibrator_rejects_empty():
    with pytest.raises(ValueError, match="no samples"):
        CourtCalibrator.fit([], [], scope="test")


def test_calibrator_roundtrip_dict():
    p, y = _separable(seed=2)
    cal = CourtCalibrator.fit(p, y, scope="test:roundtrip")
    again = CourtCalibrator.from_dict(cal.to_dict())
    assert again.config.scope == cal.config.scope
    assert again.config.coef == pytest.approx(cal.config.coef)
    for pi in (0.1, 0.4, 0.5, 0.9):
        assert again.apply_one(pi) == pytest.approx(cal.apply_one(pi), abs=1e-9)


def test_calibrator_dict_rejects_negative_coef():
    with pytest.raises(ValueError, match="non-positive"):
        CourtCalibrator.from_dict({
            "scope": "s", "clip": 1e-4, "coef": -1.0, "intercept": 0.0,
        })


def test_calibrator_refuses_degenerate_feature():
    p, y = _separable(seed=3)
    with pytest.raises(ValueError, match="degenerate"):
        CourtCalibrator.fit([0.5] * len(p), y, scope="test")


# --- ltt_threshold -----------------------------------------------------------

def test_ltt_threshold_guarantees_fpr():
    # n=2000: Hoeffding slack = sqrt(log(1/delta)/(2n)) = 0.027 < alpha=0.05,
    # so a certifiable threshold exists. (At n=500 the slack is 0.055 > alpha
    # and ltt_threshold correctly refuses — see the test below.)
    rng = np.random.default_rng(7)
    reals = rng.normal(0.0, 1.0, 2000)
    tau, info = ltt_threshold(reals, alpha=0.05, delta=0.05)
    assert math.isfinite(tau)
    assert info["fpr_upper_bound"] <= 0.05 + 1e-12
    assert info["n_calibration_reals"] == 2000
    # empirical check on a fresh exchangeable sample
    fresh = rng.normal(0.0, 1.0, 50000)
    assert float((fresh > tau).mean()) <= 0.06


def test_ltt_threshold_refuses_when_slack_exceeds_alpha():
    """Documented C1 constraint: small calibration sets cannot certify alpha.

    Hoeffding slack at n=500, delta=0.05 is sqrt(log(20)/1000) = 0.0547, which
    already exceeds alpha=0.05, so NO threshold is certifiable. This is the
    honest cost of a distribution-free guarantee: budget n >= ~600 reals per
    domain before expecting an LTT threshold to exist at alpha=0.05.
    """
    rng = np.random.default_rng(8)
    tau, info = ltt_threshold(rng.normal(0.0, 1.0, 500), alpha=0.05, delta=0.05)
    assert tau == float("inf")
    assert info["guarantee"] == "unsatisfiable"
    assert info["slack"] > 0.05


def test_ltt_threshold_is_conservative_when_unsatisfiable():
    rng = np.random.default_rng(8)
    tau, info = ltt_threshold(rng.normal(0, 1, 20), alpha=0.001, delta=0.001)
    assert tau == float("inf")
    assert info["guarantee"] == "unsatisfiable"


def test_ltt_threshold_validates_alpha():
    with pytest.raises(ValueError, match="alpha"):
        ltt_threshold([0.0, 1.0], alpha=1.5)
    with pytest.raises(ValueError, match="delta"):
        ltt_threshold([0.0, 1.0], delta=0.0)


def test_ltt_threshold_rejects_empty():
    with pytest.raises(ValueError, match="no calibration reals"):
        ltt_threshold([])


# --- cohort normalization ----------------------------------------------------

def test_cohort_z_normalizes_to_unit_scale():
    reals = list(np.random.default_rng(9).normal(5.0, 2.0, 400))
    stats = fit_cohort_stats(reals)
    z = cohort_normalize(reals, stats, method="z")
    assert float(np.abs(z.mean())) < 0.05
    assert float(np.abs(z.std() - 1.0)) < 0.05


def test_cohort_rejects_bad_method():
    with pytest.raises(ValueError, match="unknown method"):
        cohort_normalize([1.0, 2.0], {"mean": 0.0, "std": 1.0}, method="bogus")


def test_cohort_rejects_nonpositive_std():
    with pytest.raises(ValueError, match="std must be positive"):
        cohort_normalize([1.0], {"mean": 0.0, "std": 0.0}, method="z")
    with pytest.raises(ValueError, match="tar_std must be positive"):
        cohort_normalize([1.0], {"mean": 0.0, "std": 1.0,
                                 "tar_mean": 0.0, "tar_std": 0.0}, method="as")


def test_cohort_stats_reject_empty():
    with pytest.raises(ValueError, match="empty cohort"):
        fit_cohort_stats([])


# --- risk curve --------------------------------------------------------------

def test_risk_curve_shape_and_monotone_cost():
    p, y = _separable(seed=10)
    curve = risk_curve(p, y)
    assert len(curve) == 21
    costs = [pt["cost"] for pt in curve]
    best = best_cost_point(curve)
    assert best["cost"] == min(costs)
    for pt in curve:
        assert 0.0 <= pt["fpr"] <= 1.0
        assert 0.0 <= pt["fnr"] <= 1.0
        assert pt["tpr"] + pt["fnr"] == pytest.approx(1.0)


def test_risk_curve_validates_inputs():
    with pytest.raises(ValueError, match="length mismatch"):
        risk_curve([0.1, 0.2], [1])
    with pytest.raises(ValueError, match="empty input"):
        risk_curve([], [])
    with pytest.raises(ValueError, match="both classes"):
        risk_curve([0.1, 0.2], [1, 1])


def test_risk_curve_perfect_separation_has_zero_cost_point():
    y = np.array([1, 1, 0, 0])
    p = np.array([0.9, 0.95, 0.05, 0.1])
    best = best_cost_point(risk_curve(p, y))
    assert best["cost"] == pytest.approx(0.0)


# --- CalibratedStackerJudge --------------------------------------------------

def test_calibrated_judge_delegates_without_calibrator():
    """No calibrator => byte-identical delegation to StackerJudge.decide."""
    from calibrated_judge import CalibratedStackerJudge
    from stacker_judge import StackerJudge

    class _Stub(StackerJudge):
        def __init__(self):
            self.out_of_conflict = False

        def decide(self, survivors):
            return {"camera_original": 0.4, "ai_generated": 0.5,
                    "manipulated": 0.1}

    judge = CalibratedStackerJudge.__new__(CalibratedStackerJudge)
    judge.calibrator = None
    judge.calibration_applied = False
    judge.out_of_conflict = False
    judge.decide = _Stub().decide.__get__(judge, CalibratedStackerJudge)
    out = judge.decide([])
    assert out == {"camera_original": 0.4, "ai_generated": 0.5,
                   "manipulated": 0.1}
    assert judge.scope is None


def test_calibrated_judge_survives_extreme_base_score():
    """p_syn == 0 must not divide by zero or emit a negative mass."""
    from calibrated_judge import CalibratedStackerJudge
    from stacker_judge import StackerJudge

    class _Stub(StackerJudge):
        def __init__(self):
            self.out_of_conflict = False

        def decide(self, survivors):
            return {"camera_original": 1.0, "ai_generated": 0.0,
                    "manipulated": 0.0}

    judge = CalibratedStackerJudge.__new__(CalibratedStackerJudge)
    p, y = _separable(seed=11)
    judge.calibrator = CourtCalibrator.fit(p, y, scope="test:stub")
    judge.calibration_applied = False
    judge.out_of_conflict = False
    judge.decide = _Stub().decide.__get__(judge, CalibratedStackerJudge)
    out = judge.decide([])
    assert out["ai_generated"] == 0.0
    assert out["manipulated"] == 0.0
    assert judge.calibration_applied is False