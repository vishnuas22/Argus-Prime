"""Court-level score calibration (TASK-C1, layer 1-4).

Why this module exists (measured, not assumed)
--------------------------------------------
The court's channels keep AUC 0.71-0.78 cross-domain, but fixed thresholds
collapse to FPR 41-71% off-manifold (D-103 DINOv3 pre-flight, measured on this
repo). Published 2026 evidence confirms the mechanism is score-scale
misalignment, not ranking loss: monotone transforms (z-norm, temperature)
provably cannot change AUC, but they do move operating points.

What this module provides
-------------------------
1. ``logit_transform``        - probability -> log-odds (removes the documented
                                probability-scale squashing).
2. ``CourtCalibrator``        - one logistic regression on held-out fused scores,
                                turning the court score into a calibrated LLR so a
                                single global threshold is meaningful.
3. ``cohort_normalize``       - class-pure per-domain Z / AS normalisation.
4. ``ltt_threshold``          - Learn-then-Test threshold selection with a
                                finite-sample FPR <= alpha guarantee.
5. ``risk_curve``             - a-DCF-style cost-vs-threshold curve for the API.

Hard limits (repeat in every status report)
-------------------------------------------
* Calibration NEVER improves AUC. It is a monotone map on a monotone feature; the
  ranking is invariant by construction. ``CourtCalibrator`` refuses to fit a
  model with a non-positive coefficient rather than silently inverting ranking.
* LTT guarantees hold only where the calibration reals are exchangeable with
  deployment. A fully novel generator gets NO guarantee; the existing APS
  abstention band remains the only defence.
* Cohort packs need domain-matched bona-fide data. A brand-new platform codec
  needs a new cohort before its scores are trustworthy.

Vendor boundary (D-009)
-----------------------
Nothing here edits ``integrations/Argus_Master/``. ``fusion.py`` keeps its
pristine ``sigmoid(3.0 * fused)`` squash; this module is harness-level and is
injected through the existing pluggable-judge seam.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np

DEFAULT_CLIP = 1e-4
VALID_COHORT_METHODS = ("z", "as")


def logit_transform(p, clip: float = DEFAULT_CLIP) -> np.ndarray:
    """Map probabilities to log-odds.

    Parameters
    ----------
    p : float or array-like
        Probability-like score in [0, 1].
    clip : float
        Clamp applied before the transform so 0 and 1 stay finite.

    Returns
    -------
    numpy.ndarray
        ``log(p / (1 - p))``, monotone increasing in ``p``.
    """
    arr = np.asarray(p, dtype=float)
    arr = np.clip(arr, clip, 1.0 - clip)
    return np.log(arr / (1.0 - arr))


def sigmoid(x) -> np.ndarray:
    """Numerically stable logistic function (inverse of ``logit_transform``)."""
    arr = np.asarray(x, dtype=float)
    out = np.empty_like(arr)
    pos = arr >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-arr[pos]))
    ex = np.exp(arr[~pos])
    out[~pos] = ex / (1.0 + ex)
    return out


@dataclass
class CalibrationConfig:
    """Serialisable description of a fitted calibration, with its scope label.

    ``scope`` is mandatory (D-037: no number without a scope label). It records
    which cohort/domain the calibration was fitted on so a consumer can tell
    whether the guarantee transfers.
    """

    scope: str
    method: str = "logistic"
    clip: float = DEFAULT_CLIP
    coef: float = 0.0
    intercept: float = 0.0
    n_train: int = 0
    n_positive: int = 0
    base_auc: float | None = None
    meta: dict = field(default_factory=dict)

    @property
    def is_monotone_increasing(self) -> bool:
        """True when the calibrator preserves the base ranking direction."""
        return self.coef > 0.0


class CourtCalibrator:
    """Logistic-regression calibration of the court's fused score.

    The feature is ``logit(p_syn)`` so the map is monotone in the base score and
    therefore AUC-invariant. Fit fails closed if the learned coefficient is not
    positive: an inverted coefficient would silently flip the ranking, which is
    worse than no calibration at all.
    """

    def __init__(self, config: CalibrationConfig | None = None):
        self.config = config
        self._model = None

    @classmethod
    def fit(
        cls,
        p_syn: Sequence[float],
        y_fake: Sequence[int],
        scope: str,
        clip: float = DEFAULT_CLIP,
        C: float = 1.0,
    ) -> "CourtCalibrator":
        """Fit the calibrator on held-out fused scores.

        Parameters
        ----------
        p_syn : sequence of float
            Base court probability that the input is synthetic.
        y_fake : sequence of int
            1 for fake, 0 for real. Must contain both classes.
        scope : str
            D-037 scope label describing the calibration cohort.
        clip : float
            Clamp for the logit transform.
        C : float
            Inverse regularisation strength. Low sample sizes need regularisation;
            the default keeps the map close to the identity in log-odds space.

        Raises
        ------
        ValueError
            If the labels are single-class, inputs are empty, or the fitted
            coefficient is non-positive.
        """
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import roc_auc_score

        x = logit_transform(p_syn, clip=clip).reshape(-1, 1)
        y = np.asarray(y_fake, dtype=int).reshape(-1)
        if x.shape[0] == 0:
            raise ValueError("CourtCalibrator.fit: no samples")
        if y.min() == y.max():
            raise ValueError(
                "CourtCalibrator.fit: single-class labels — calibration is not "
                "identifiable and would encode the class prior, not a score map"
            )
        if len(np.unique(x)) < 2:
            raise ValueError("CourtCalibrator.fit: degenerate feature (all equal)")

        model = LogisticRegression(C=C, solver="lbfgs", max_iter=1000)
        model.fit(x, y)
        coef = float(model.coef_[0, 0])
        if coef <= 0.0:
            raise ValueError(
                f"CourtCalibrator.fit: non-positive coefficient {coef:.6g} would "
                "invert the ranking (AUC -> 1 - AUC); refusing to fit"
            )

        base_auc = float(roc_auc_score(y, x))
        cal_p = model.predict_proba(x)[:, 1]
        cal_auc = float(roc_auc_score(y, cal_p))
        if abs(cal_auc - base_auc) > 1e-9:
            raise ValueError(
                f"CourtCalibrator.fit: AUC moved {base_auc:.6f} -> {cal_auc:.6f}; "
                "a monotone map must preserve ranking"
            )

        cfg = CalibrationConfig(
            scope=scope,
            method="logistic",
            clip=clip,
            coef=coef,
            intercept=float(model.intercept_[0]),
            n_train=int(y.shape[0]),
            n_positive=int(y.sum()),
            base_auc=base_auc,
            meta={"C": C, "calibrated_auc": cal_auc},
        )
        cal = cls(cfg)
        cal._model = model
        return cal

    def transform(self, p_syn) -> np.ndarray:
        """Map base synthetic probabilities to calibrated probabilities."""
        if self._model is None:
            raise RuntimeError("CourtCalibrator is not fitted")
        x = logit_transform(p_syn, clip=self.config.clip).reshape(-1, 1)
        return self._model.predict_proba(x)[:, 1]

    def apply_one(self, p_syn: float) -> float:
        """Calibrate a single probability."""
        return float(self.transform([p_syn])[0])

    def to_dict(self) -> dict:
        """Plain-dict form for joblib/JSON persistence."""
        return {
            "scope": self.config.scope,
            "method": self.config.method,
            "clip": self.config.clip,
            "coef": self.config.coef,
            "intercept": self.config.intercept,
            "n_train": self.config.n_train,
            "n_positive": self.config.n_positive,
            "base_auc": self.config.base_auc,
            "meta": self.config.meta,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "CourtCalibrator":
        """Rebuild from :meth:`to_dict` output (no sklearn object required)."""
        cfg = CalibrationConfig(
            scope=d["scope"],
            method=d.get("method", "logistic"),
            clip=float(d.get("clip", DEFAULT_CLIP)),
            coef=float(d["coef"]),
            intercept=float(d["intercept"]),
            n_train=int(d.get("n_train", 0)),
            n_positive=int(d.get("n_positive", 0)),
            base_auc=d.get("base_auc"),
            meta=dict(d.get("meta") or {}),
        )
        if cfg.coef <= 0.0:
            raise ValueError("CourtCalibrator.from_dict: non-positive coefficient")
        cal = cls(cfg)
        cal._model = _LinearLogOdds(cfg)
        return cal


class _LinearLogOdds:
    """Minimal stand-in for LogisticRegression so a persisted config stays usable.

    ``CourtCalibrator`` only ever calls ``predict_proba`` on a single logit
    feature, so the fitted affine map is sufficient and avoids a hard sklearn
    dependency at serve time.
    """

    def __init__(self, cfg: CalibrationConfig):
        self._cfg = cfg

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        z = self._cfg.coef * np.asarray(x, dtype=float)[:, 0] + self._cfg.intercept
        p = sigmoid(z)
        return np.column_stack([1.0 - p, p])


def cohort_normalize(
    scores: Sequence[float],
    cohort_stats: dict,
    method: str = "z",
) -> np.ndarray:
    """Normalise scores against a class-pure bona-fide cohort.

    Parameters
    ----------
    scores : sequence of float
        Scores to normalise (any class).
    cohort_stats : dict
        Must contain ``mean`` and ``std`` (Z-norm) or ``mean`` plus a target
        cohort offset/scale for AS-norm (``mean``, ``std``, ``tar_mean``,
        ``tar_std``). Cohort stats MUST be computed on bona-fide data only —
        a contaminated cohort provably makes transfer worse (arXiv 2606.21584).
    method : {"z", "as"}
        Z-normalisation or adaptive symmetric normalisation.

    Returns
    -------
    numpy.ndarray
        Normalised scores.
    """
    if method not in VALID_COHORT_METHODS:
        raise ValueError(f"cohort_normalize: unknown method {method!r}")
    s = np.asarray(scores, dtype=float)
    mean = float(cohort_stats["mean"])
    std = float(cohort_stats["std"])
    if std <= 0.0:
        raise ValueError("cohort_normalize: cohort std must be positive")
    if method == "z":
        return (s - mean) / std
    tar_mean = float(cohort_stats["tar_mean"])
    tar_std = float(cohort_stats["tar_std"])
    if tar_std <= 0.0:
        raise ValueError("cohort_normalize: AS-norm tar_std must be positive")
    return (s - mean) / (tar_std + 1e-9) + tar_mean


def fit_cohort_stats(scores: Sequence[float]) -> dict:
    """Compute class-pure cohort statistics from bona-fide scores only.

    Provided so callers build cohorts correctly by construction rather than by
    remembering to filter.
    """
    s = np.asarray(scores, dtype=float)
    if s.size == 0:
        raise ValueError("fit_cohort_stats: empty cohort")
    return {
        "mean": float(s.mean()),
        "std": float(s.std(ddof=0)),
        "n": int(s.size),
        "scope": "bona-fide-only",
    }


def ltt_threshold(
    real_scores: Sequence[float],
    alpha: float = 0.05,
    delta: float = 0.05,
    candidates: Iterable[float] | None = None,
) -> tuple[float, dict]:
    """Learn-then-Test threshold with a finite-sample FPR guarantee.

    Selects the smallest threshold ``tau`` (maximising catch) such that the
    upper confidence bound on the false-positive rate over the calibration
    reals is at most ``alpha``, with confidence ``1 - delta``. Uses a Hoeffding
    bound, which is distribution-free and valid for any exchangeable calibration
    sample.

    Convention: a clip is flagged fake when ``score > tau``. Higher ``tau``
    means fewer flags and a lower FPR, so the smallest admissible ``tau`` is the
    most sensitive valid operating point.

    Sample-size constraint (measured, do not skip)
    ------------------------------------------------
    The Hoeffding slack is ``sqrt(log(1/delta) / (2n))`` and must be ``< alpha``
    for any threshold to be certifiable. At ``delta=0.05`` this requires
    ``n >= log(1/delta) / (2 * alpha**2)``: for ``alpha=0.05`` that is
    **n >= ~600 bona-fide reals per domain**, and for ``alpha=0.01`` it is
    **n >= ~15,000**. Below the threshold the function correctly returns
    ``(inf, {"guarantee": "unsatisfiable"})`` rather than pretending to certify.
    Budget calibration data accordingly; a 500-sample cohort cannot certify a
    5% FPR at 95% confidence.

    Parameters
    ----------
    real_scores : sequence of float
        Scores for bona-fide (real) calibration samples ONLY.
    alpha : float
        Target upper bound on FPR.
    delta : float
        Allowed failure probability of the guarantee.
    candidates : iterable of float, optional
        Threshold grid. Defaults to the observed real scores.

    Returns
    -------
    (tau, info) : (float, dict)
        ``info`` carries the empirical FPR, the bound, and the sample count so
        the decision is auditable.
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError("ltt_threshold: alpha must be in (0, 1)")
    if not 0.0 < delta < 1.0:
        raise ValueError("ltt_threshold: delta must be in (0, 1)")
    reals = np.asarray(real_scores, dtype=float)
    if reals.size == 0:
        raise ValueError("ltt_threshold: no calibration reals supplied")
    if candidates is None:
        grid = np.unique(reals)
    else:
        grid = np.asarray(sorted(float(c) for c in candidates), dtype=float)
    if grid.size == 0:
        raise ValueError("ltt_threshold: empty candidate grid")

    n = reals.size
    slack = math.sqrt(math.log(1.0 / delta) / (2.0 * n))

    best_tau = None
    best_info = None
    for tau in grid:
        emp = float((reals > tau).mean())
        ucb = emp + slack
        if ucb <= alpha:
            best_tau = float(tau)
            best_info = {
                "empirical_fpr": emp,
                "fpr_upper_bound": ucb,
                "n_calibration_reals": n,
                "alpha": alpha,
                "delta": delta,
                "slack": slack,
                "guarantee": f"FPR <= {alpha} with confidence >= {1 - delta}",
            }
            break

    if best_tau is None:
        above = float((reals > grid.max()).mean())
        return float("inf"), {
            "empirical_fpr": above,
            "fpr_upper_bound": above + slack,
            "n_calibration_reals": n,
            "alpha": alpha,
            "delta": delta,
            "slack": slack,
            "guarantee": "unsatisfiable",
            "reason": (
                "no candidate threshold admits the FPR bound; the calibration "
                "sample is too small or the score does not separate"
            ),
        }

    return best_tau, best_info


def risk_curve(
    scores: Sequence[float],
    labels: Sequence[int],
    thresholds: Sequence[float] | None = None,
    cost_fa: float = 1.0,
    cost_miss: float = 1.0,
) -> list[dict]:
    """a-DCF-style cost-vs-threshold curve.

    Parameters
    ----------
    scores : sequence of float
        Detector scores (higher = more fake).
    labels : sequence of int
        1 for fake, 0 for real.
    thresholds : sequence of float, optional
        Grid. Defaults to the observed score range.
    cost_fa, cost_miss : float
        Relative cost of a false alarm vs a missed detection.

    Returns
    -------
    list of dict
        ``[{"threshold": t, "fpr": .., "fnr": .., "tpr": .., "cost": ..}, ...]``
        sorted by threshold. The curve is monotone in cost by construction.
    """
    s = np.asarray(scores, dtype=float)
    y = np.asarray(labels, dtype=int)
    if s.shape != y.shape:
        raise ValueError("risk_curve: scores/labels length mismatch")
    if s.size == 0:
        raise ValueError("risk_curve: empty input")
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    if n_pos == 0 or n_neg == 0:
        raise ValueError("risk_curve: need both classes to compute FPR/FNR")

    if thresholds is None:
        lo, hi = float(s.min()), float(s.max())
        grid = np.linspace(lo, hi, 21)
    else:
        grid = np.asarray(sorted(float(t) for t in thresholds), dtype=float)

    out = []
    for tau in grid:
        flagged = s > tau
        tp = float((flagged & (y == 1)).sum())
        fp = float((flagged & (y == 0)).sum())
        tpr = tp / n_pos
        fpr = fp / n_neg
        fnr = 1.0 - tpr
        cost = cost_fa * fpr + cost_miss * fnr
        out.append({
            "threshold": round(float(tau), 6),
            "fpr": round(fpr, 6),
            "fnr": round(fnr, 6),
            "tpr": round(tpr, 6),
            "cost": round(cost, 6),
        })
    return out


def best_cost_point(curve: Sequence[dict]) -> dict:
    """Return the minimum-cost point of a :func:`risk_curve`."""
    if not curve:
        raise ValueError("best_cost_point: empty curve")
    return min(curve, key=lambda p: p["cost"])