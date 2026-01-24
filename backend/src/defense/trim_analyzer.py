"""
TRIM Analyzer - Training-free Adversarial Detection.

Implements TRIM (Training-free Robust detection via Information Maximization)
for detecting adversarial examples without retraining.

Theory:
- Adversarial examples shift prediction entropy
- KL divergence between clean and perturbed predictions
- No retraining required
- Defends against FGSM/BIM/DeepFool/PGD attacks
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats
from scipy.spatial.distance import jensenshannon

from src.config import get_config
from src.core.logging import logger


@dataclass
class TRIMResult:
    """Result from TRIM adversarial analysis."""
    is_adversarial: bool
    entropy_score: float
    kl_divergence: float
    js_divergence: float
    confidence: float
    details: Dict[str, Any] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "is_adversarial": self.is_adversarial,
            "entropy_score": self.entropy_score,
            "kl_divergence": self.kl_divergence,
            "js_divergence": self.js_divergence,
            "confidence": self.confidence,
            "details": self.details or {},
        }


class TRIMAnalyzer:
    """
    TRIM-based Adversarial Example Detection.

    Uses information-theoretic measures to detect adversarial perturbations:
    - Prediction entropy analysis
    - KL divergence between clean and perturbed predictions
    - Jensen-Shannon divergence for distribution comparison

    Attributes:
        threshold: Threshold for adversarial detection
        entropy_window: Window size for entropy smoothing
        device: Computation device
    """

    def __init__(
        self,
        threshold: float = 0.1,
        entropy_window: int = 5,
        device: Optional[torch.device] = None,
    ):
        """
        Initialize TRIM analyzer.

        Args:
            threshold: Threshold for adversarial detection
            entropy_window: Window size for entropy smoothing
            device: Computation device
        """
        self.threshold = threshold
        self.entropy_window = entropy_window
        self.device = device or torch.device("cpu")

        config = get_config()
        self.defense_config = config.defense

        logger.info(f"TRIMAnalyzer initialized: threshold={threshold}")

    def compute_entropy(self, predictions: torch.Tensor) -> torch.Tensor:
        """
        Compute prediction entropy.

        Args:
            predictions: Model predictions (logits or probabilities)

        Returns:
            Entropy values
        """
        probs = F.softmax(predictions, dim=-1)
        probs = probs + 1e-10  # Avoid log(0)
        entropy = -torch.sum(probs * torch.log(probs), dim=-1)
        return entropy

    def compute_kl_divergence(
        self,
        clean_probs: torch.Tensor,
        perturbed_probs: torch.Tensor,
    ) -> float:
        """
        Compute KL divergence between clean and perturbed predictions.

        Args:
            clean_probs: Predictions on clean input
            perturbed_probs: Predictions on perturbed input

        Returns:
            KL divergence value
        """
        clean_probs = clean_probs + 1e-10
        perturbed_probs = perturbed_probs + 1e-10

        clean_probs = clean_probs / clean_probs.sum(dim=-1, keepdim=True)
        perturbed_probs = perturbed_probs / perturbed_probs.sum(dim=-1, keepdim=True)

        kl = F.kl_div(
            torch.log(perturbed_probs),
            clean_probs,
            reduction="batchmean",
            log_target=False,
        )

        return kl.item()

    def compute_js_divergence(
        self,
        clean_probs: torch.Tensor,
        perturbed_probs: torch.Tensor,
    ) -> float:
        """
        Compute Jensen-Shannon divergence.

        Args:
            clean_probs: Clean prediction probabilities
            perturbed_probs: Perturbed prediction probabilities

        Returns:
            JS divergence value
        """
        clean_probs = clean_probs.detach().cpu().numpy()
        perturbed_probs = perturbed_probs.detach().cpu().numpy()

        if clean_probs.ndim > 1:
            clean_probs = clean_probs.mean(axis=0)
        if perturbed_probs.ndim > 1:
            perturbed_probs = perturbed_probs.mean(axis=0)

        clean_probs = clean_probs + 1e-10
        perturbed_probs = perturbed_probs / perturbed_probs.sum()

        perturbed_probs = perturbed_probs + 1e-10
        clean_probs = clean_probs / clean_probs.sum()

        m = 0.5 * (clean_probs + perturbed_probs)

        js = 0.5 * stats.entropy(clean_probs, m) + 0.5 * stats.entropy(perturbed_probs, m)

        return float(js)

    def analyze_predictions(
        self,
        clean_logits: torch.Tensor,
        perturbed_logits: torch.Tensor,
    ) -> TRIMResult:
        """
        Analyze if input is adversarial using TRIM.

        Args:
            clean_logits: Model predictions on clean input
            perturbed_logits: Model predictions on perturbed input

        Returns:
            TRIMResult with adversarial detection results
        """
        clean_probs = F.softmax(clean_logits, dim=-1)
        perturbed_probs = F.softmax(perturbed_logits, dim=-1)

        clean_entropy = self.compute_entropy(clean_logits)
        perturbed_entropy = self.compute_entropy(perturbed_logits)

        clean_entropy_val = clean_entropy.mean().item() if clean_entropy.numel() > 1 else clean_entropy.item()
        perturbed_entropy_val = perturbed_entropy.mean().item() if perturbed_entropy.numel() > 1 else perturbed_entropy.item()

        entropy_change = abs(perturbed_entropy_val - clean_entropy_val)

        kl_div = self.compute_kl_divergence(clean_probs, perturbed_probs)
        js_div = self.compute_js_divergence(clean_probs, perturbed_probs)

        combined_score = (
            0.4 * entropy_change + 0.3 * kl_div + 0.3 * js_div
        )

        is_adversarial = combined_score > self.threshold

        confidence = min(combined_score * 5, 1.0)

        details = {
            "clean_entropy": clean_entropy_val,
            "perturbed_entropy": perturbed_entropy_val,
            "entropy_change": entropy_change,
            "threshold": self.threshold,
            "combined_score": combined_score,
        }

        return TRIMResult(
            is_adversarial=is_adversarial,
            entropy_score=entropy_change,
            kl_divergence=kl_div,
            js_divergence=js_div,
            confidence=confidence,
            details=details,
        )

    def analyze_single(
        self,
        predictions: torch.Tensor,
        reference_predictions: Optional[torch.Tensor] = None,
    ) -> TRIMResult:
        """
        Analyze single prediction for adversarial patterns.

        Args:
            predictions: Model predictions
            reference_predictions: Reference predictions for comparison

        Returns:
            TRIMResult with analysis
        """
        if reference_predictions is not None:
            return self.analyze_predictions(reference_predictions, predictions)

        entropy = self.compute_entropy(predictions)
        mean_entropy = entropy.mean().item()

        is_adversarial = mean_entropy < 0.1 or mean_entropy > 4.5

        details = {
            "mean_entropy": mean_entropy,
            "std_entropy": entropy.std().item(),
            "min_entropy": entropy.min().item(),
            "max_entropy": entropy.max().item(),
        }

        confidence = 0.5

        return TRIMResult(
            is_adversarial=is_adversarial,
            entropy_score=mean_entropy,
            kl_divergence=0.0,
            js_divergence=0.0,
            confidence=confidence,
            details=details,
        )

    def batch_analyze(
        self,
        predictions: torch.Tensor,
        window_size: int = 10,
    ) -> TRIMResult:
        """
        Analyze batch predictions for adversarial patterns.

        Args:
            predictions: Batch of predictions
            window_size: Window size for smoothing

        Returns:
            TRIMResult with batch analysis
        """
        entropy = self.compute_entropy(predictions)

        if len(entropy) < window_size:
            return self.analyze_single(predictions)

        smoothed_entropy = []
        for i in range(len(entropy)):
            start = max(0, i - window_size // 2)
            end = min(len(entropy), i + window_size // 2)
            smoothed_entropy.append(entropy[start:end].mean().item())

        smoothed_entropy = torch.tensor(smoothed_entropy)
        entropy_change = torch.std(smoothed_entropy).item()

        is_adversarial = entropy_change > 0.5 or torch.mean(smoothed_entropy).item() < 0.2

        confidence = min(entropy_change * 2, 1.0)

        details = {
            "mean_entropy": torch.mean(smoothed_entropy).item(),
            "std_entropy": entropy_change,
            "window_size": window_size,
        }

        return TRIMResult(
            is_adversarial=is_adversarial,
            entropy_score=entropy_change,
            kl_divergence=0.0,
            js_divergence=0.0,
            confidence=confidence,
            details=details,
        )


def create_trim_analyzer() -> TRIMAnalyzer:
    """Factory function for TRIMAnalyzer."""
    config = get_config()
    return TRIMAnalyzer(
        threshold=config.defense.trim_threshold,
    )
