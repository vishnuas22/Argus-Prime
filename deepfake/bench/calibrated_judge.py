"""CalibratedStackerJudge — TASK-C1 court-level calibration, harness-side.

Why a subclass and not an edit
------------------------------
``integrations/Argus_Master/backend/fusion.py`` is VENDORED (D-009: keep
pristine). Its ``fuse()`` already accepts a pluggable ``judge`` and the server
already builds ``StackerJudge`` at ``deepfake/api/server.py:192`` and injects it.
Subclassing here means calibration lands with ZERO vendored edits and the default
path stays byte-identical when the feature flag is off.

What it changes
---------------
``StackerJudge.decide`` returns three hypothesis probabilities derived from the
stacker's ``predict_proba``. This wrapper:
  1. computes the base probabilities exactly as before,
  2. maps the synthetic mass through a fitted ``CourtCalibrator`` (monotone,
     therefore AUC-invariant),
  3. re-splits the calibrated synthetic mass into ai_generated / manipulated
     using the SAME localisation ratio as the base judge,
  4. leaves ``out_of_conflict`` untouched (advisory, non-destructive).

Because the conformal set is computed from the returned probabilities, the
abstention decision inherits the calibrated scale automatically.

Fail-closed: if no calibrator is supplied, every method delegates to the base
implementation, so an unlabelled deployment cannot silently change verdicts.
"""
from __future__ import annotations

from stacker_judge import StackerJudge


class CalibratedStackerJudge(StackerJudge):
    """StackerJudge with a monotone court-level probability calibration."""

    name = "histgbm-stack-v1+cal"

    def __init__(self, *args, calibrator=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.calibrator = calibrator
        self.calibration_applied = False

    def decide(self, survivors):
        base = super().decide(survivors)
        if self.calibrator is None:
            self.calibration_applied = False
            return base

        p_syn_base = float(base["ai_generated"]) + float(base["manipulated"])
        if p_syn_base <= 0.0:
            self.calibration_applied = False
            return base

        manip_ratio = float(base["manipulated"]) / p_syn_base
        p_syn_cal = float(self.calibrator.apply_one(p_syn_base))

        p_manip = p_syn_cal * manip_ratio
        p_ai = p_syn_cal - p_manip
        self.calibration_applied = True
        return {
            "camera_original": round(1.0 - p_syn_cal, 4),
            "ai_generated": round(p_ai, 4),
            "manipulated": round(p_manip, 4),
        }

    @property
    def scope(self) -> str | None:
        """D-037 scope label of the fitted calibration, or None when disabled."""
        if self.calibrator is None:
            return None
        return getattr(self.calibrator.config, "scope", None)