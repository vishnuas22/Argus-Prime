"""
Localization Generator - Temporal and Spatial Manipulation Detection.

Localizes manipulation regions in time and space for forensic evidence
with attention-based temporal localization and mask prediction.

Algorithms:
- Temporal: Frame-level confidence scores with peak detection
- Spatial: Heatmap thresholding with contour detection
- Smoothing: Savitzky-Golay filter for noise reduction
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import cv2
from scipy import signal
from scipy.ndimage import gaussian_filter
from scipy.signal import find_peaks, savgol_filter

from src.config import get_config
from src.core.logging import logger


@dataclass
class TemporalRegion:
    """Temporal manipulation region."""
    start_frame: int
    end_frame: int
    confidence: float
    peak_frame: Optional[int] = None
    duration_frames: int = 0


@dataclass
class SpatialRegion:
    """Spatial manipulation region."""
    mask: np.ndarray
    bbox: List[int]
    centroid: Tuple[int, int]
    area: int
    confidence: float
    polygon: List[Tuple[int, int]] = field(default_factory=list)


@dataclass
class LocalizationResult:
    """Result from localization analysis."""
    temporal_regions: List[TemporalRegion]
    spatial_regions: List[SpatialRegion]
    frame_timestamps: List[Dict[str, Any]]
    overall_manipulation_mask: Optional[np.ndarray] = None
    confidence: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "temporal_regions": [
                {
                    "start_frame": r.start_frame,
                    "end_frame": r.end_frame,
                    "confidence": r.confidence,
                    "peak_frame": r.peak_frame,
                    "duration_frames": r.duration_frames,
                }
                for r in self.temporal_regions
            ],
            "spatial_regions": [
                {
                    "bbox": r.bbox,
                    "centroid": list(r.centroid),
                    "area": r.area,
                    "confidence": r.confidence,
                    "polygon": r.polygon,
                }
                for r in self.spatial_regions
            ],
            "frame_timestamps": self.frame_timestamps,
            "confidence": self.confidence,
        }


class Localizer:
    """
    Localization Generator for Manipulation Detection.

    Provides temporal and spatial localization of manipulation regions:
    - Temporal: Frame-level timestamps via peak detection
    - Spatial: Pixel-level masks via heatmap thresholding
    - Combined: Unified localization result

    Attributes:
        peak_prominence: Minimum prominence for peak detection
        peak_distance: Minimum distance between peaks
        smoothing_window: Window size for Savitzky-Golay filter
        mask_threshold: Threshold for binary mask generation
        min_region_area: Minimum contour area for region detection
    """

    def __init__(
        self,
        peak_prominence: float = 0.3,
        peak_distance: int = 5,
        smoothing_window: int = 11,
        mask_threshold: float = 0.5,
        min_region_area: int = 100,
        use_gpu: bool = True,
    ):
        """
        Initialize localization generator.

        Args:
            peak_prominence: Minimum prominence for peak detection
            peak_distance: Minimum distance between peaks
            smoothing_window: Window size for Savitzky-Golay filter
            mask_threshold: Threshold for binary mask generation
            min_region_area: Minimum contour area for region detection
            use_gpu: Whether to use GPU acceleration
        """
        self.peak_prominence = peak_prominence
        self.peak_distance = peak_distance
        self.smoothing_window = smoothing_window
        self.mask_threshold = mask_threshold
        self.min_region_area = min_region_area
        self.device = "cuda" if use_gpu and False else "cpu"

        config = get_config()
        self.fps = config.analysis.default_fps

        logger.info(
            f"Localizer initialized: peak_prominence={peak_prominence}, "
            f"mask_threshold={mask_threshold}"
        )

    def _smooth_scores(self, scores: List[float]) -> np.ndarray:
        """Apply Savitzky-Golay smoothing to scores."""
        scores_array = np.array(scores)

        if len(scores_array) < self.smoothing_window:
            return scores_array

        try:
            smoothed = savgol_filter(
                scores_array,
                window_length=min(self.smoothing_window, len(scores_array) - 1),
                polyorder=3,
            )
            return smoothed
        except Exception as e:
            logger.warning(f"Smoothing failed: {e}")
            return scores_array

    def _detect_peaks(
        self,
        scores: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect peaks in manipulation scores.

        Args:
            scores: Array of manipulation scores

        Returns:
            Tuple of (peak_indices, properties)
        """
        if len(scores) < self.peak_distance:
            return np.array([]), {}

        try:
            peaks, properties = find_peaks(
                scores,
                prominence=self.peak_prominence,
                distance=self.peak_distance,
            )
            return peaks, properties
        except Exception as e:
            logger.warning(f"Peak detection failed: {e}")
            return np.array([]), {}

    def _cluster_peaks(
        self,
        peaks: np.ndarray,
        scores: np.ndarray,
        max_gap: int = 10,
    ) -> List[TemporalRegion]:
        """
        Cluster nearby peaks into temporal regions.

        Args:
            peaks: Array of peak frame indices
            scores: Array of manipulation scores
            max_gap: Maximum gap between peaks in same region

        Returns:
            List of clustered temporal regions
        """
        if len(peaks) == 0:
            return []

        if len(peaks) == 1:
            peak_idx = int(peaks[0])
            if peak_idx < len(scores):
                confidence = float(scores[peak_idx])
            else:
                confidence = 0.5
            return [
                TemporalRegion(
                    start_frame=max(0, peak_idx - 5),
                    end_frame=peak_idx + 5,
                    confidence=confidence,
                    peak_frame=peak_idx,
                    duration_frames=11,
                )
            ]

        regions = []
        current_cluster = [int(peaks[0])]

        for i in range(1, len(peaks)):
            peak_val = int(peaks[i])
            if peak_val - current_cluster[-1] <= max_gap:
                current_cluster.append(peak_val)
            else:
                cluster_scores = [float(scores[p]) for p in current_cluster if p < len(scores)]
                if cluster_scores:
                    peak_idx = current_cluster[np.argmax(cluster_scores)]
                    regions.append(
                        TemporalRegion(
                            start_frame=max(0, min(current_cluster) - 5),
                            end_frame=max(current_cluster) + 5,
                            confidence=max(cluster_scores),
                            peak_frame=peak_idx,
                            duration_frames=max(current_cluster) - min(current_cluster) + 11,
                        )
                    )
                current_cluster = [peak_val]

        if current_cluster:
            cluster_scores = [float(scores[p]) for p in current_cluster if p < len(scores)]
            if cluster_scores:
                peak_idx = current_cluster[np.argmax(cluster_scores)]
                regions.append(
                    TemporalRegion(
                        start_frame=max(0, min(current_cluster) - 5),
                        end_frame=max(current_cluster) + 5,
                        confidence=max(cluster_scores),
                        peak_frame=peak_idx,
                        duration_frames=max(current_cluster) - min(current_cluster) + 11,
                    )
                )

        regions.sort(key=lambda r: r.confidence, reverse=True)

        return regions

    def localize_temporal(
        self,
        per_frame_scores: List[Dict[str, Any]],
        fps: int = 5,
    ) -> Tuple[List[TemporalRegion], List[Dict[str, Any]]]:
        """
        Localize manipulation in time.

        Args:
            per_frame_scores: List of per-frame analysis results
            fps: Frames per second for timestamp calculation

        Returns:
            Tuple of (temporal_regions, frame_timestamps)
        """
        if not per_frame_scores:
            return [], []

        scores = [
            f.get("manipulation_score", 0.0)
            for f in per_frame_scores
        ]

        smoothed_scores = self._smooth_scores(scores)

        peaks, _ = self._detect_peaks(smoothed_scores)

        temporal_regions = self._cluster_peaks(peaks, smoothed_scores)

        frame_timestamps = []
        for i, frame in enumerate(per_frame_scores):
            frame_timestamps.append({
                "frame_number": i,
                "timestamp_seconds": round(i / fps, 3),
                "manipulation_score": scores[i],
                "smoothed_score": float(smoothed_scores[i]) if i < len(smoothed_scores) else scores[i],
                "is_manipulated": scores[i] > 0.5,
            })

        if not temporal_regions and np.mean(scores) > 0.5:
            high_score_frames = [
                i for i, s in enumerate(scores) if s > 0.5
            ]
            if high_score_frames:
                temporal_regions.append(
                    TemporalRegion(
                        start_frame=max(0, min(high_score_frames) - 5),
                        end_frame=max(high_score_frames) + 5,
                        confidence=float(np.mean([scores[i] for i in high_score_frames])),
                        peak_frame=high_score_frames[np.argmax([scores[i] for i in high_score_frames])],
                        duration_frames=max(high_score_frames) - min(high_score_frames) + 11,
                    )
                )

        logger.info(
            f"Temporal localization: {len(temporal_regions)} regions detected "
            f"from {len(per_frame_scores)} frames"
        )

        return temporal_regions, frame_timestamps

    def _threshold_heatmap(
        self,
        heatmap: np.ndarray,
        method: str = "otsu",
    ) -> np.ndarray:
        """
        Threshold heatmap to binary mask.

        Args:
            heatmap: Heatmap array
            method: Thresholding method (otsu, adaptive, fixed)

        Returns:
            Binary mask
        """
        heatmap_normalized = (heatmap - heatmap.min()) / (
            heatmap.max() - heatmap.min() + 1e-8
        )

        if method == "otsu":
            _, binary = cv2.threshold(
                (heatmap_normalized * 255).astype(np.uint8),
                0,
                255,
                cv2.THRESH_BINARY + cv2.THRESH_OTSU,
            )
        elif method == "adaptive":
            binary = cv2.adaptiveThreshold(
                (heatmap_normalized * 255).astype(np.uint8),
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                11,
                2,
            )
        else:
            threshold = self.mask_threshold
            binary = np.zeros_like(heatmap_normalized, dtype=np.uint8)
            binary[(heatmap_normalized * 255).astype(np.uint8) > int(threshold * 255)] = 255

        return binary > 0

    def _detect_contours(
        self,
        binary_mask: np.ndarray,
    ) -> List[np.ndarray]:
        """
        Detect contours in binary mask.

        Args:
            binary_mask: Binary mask array

        Returns:
            List of contours
        """
        binary_uint8 = binary_mask.astype(np.uint8) * 255

        contours, _ = cv2.findContours(
            binary_uint8,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        return contours

    def _simplify_polygon(
        self,
        contour: np.ndarray,
        epsilon: float = 0.02,
    ) -> List[Tuple[int, int]]:
        """
        Simplify contour polygon.

        Args:
            contour: Contour array
            epsilon: Approximation accuracy

        Returns:
            Simplified polygon as list of points
        """
        perimeter = cv2.arcLength(contour, True)
        epsilon = epsilon * perimeter

        approx = cv2.approxPolyDP(contour, epsilon, True)

        return [(int(p[0][0]), int(p[0][1])) for p in approx]

    def localize_spatial(
        self,
        heatmap: np.ndarray,
        original_size: Tuple[int, int],
        face_bbox: Optional[List[int]] = None,
    ) -> List[SpatialRegion]:
        """
        Localize manipulation in space.

        Args:
            heatmap: Heatmap array
            original_size: Original image size (width, height)
            face_bbox: Optional face bounding box

        Returns:
            List of spatial regions
        """
        if heatmap is None or heatmap.size == 0:
            return []

        heatmap_resized = cv2.resize(heatmap, original_size)

        binary_mask = self._threshold_heatmap(heatmap_resized)

        if face_bbox:
            x1, y1, x2, y2 = face_bbox
            h, w = binary_mask.shape
            scale_x = w / original_size[0]
            scale_y = h / original_size[1]
            x1_scaled = int(x1 * scale_x)
            y1_scaled = int(y1 * scale_y)
            x2_scaled = int(x2 * scale_x)
            y2_scaled = int(y2 * scale_y)
            x1_scaled = max(0, min(x1_scaled, w))
            y1_scaled = max(0, min(y1_scaled, h))
            x2_scaled = max(0, min(x2_scaled, w))
            y2_scaled = max(0, min(y2_scaled, h))
            mask_padded = np.zeros_like(binary_mask, dtype=bool)
            mask_padded[y1_scaled:y2_scaled, x1_scaled:x2_scaled] = True
            binary_mask = binary_mask & mask_padded

        contours = self._detect_contours(binary_mask)

        regions = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.min_region_area:
                continue

            x, y, w, h = cv2.boundingRect(contour)
            centroid = (int(x + w / 2), int(y + h / 2))

            mask = np.zeros(heatmap_resized.shape, dtype=np.float32)
            cv2.drawContours(mask, [contour], -1, 1.0, -1)

            confidence = float(np.mean(heatmap_resized * mask))

            polygon = self._simplify_polygon(contour)

            region = SpatialRegion(
                mask=mask,
                bbox=[x, y, x + w, y + h],
                centroid=centroid,
                area=int(area),
                confidence=confidence,
                polygon=polygon,
            )
            regions.append(region)

        regions.sort(key=lambda r: r.confidence, reverse=True)

        logger.info(
            f"Spatial localization: {len(regions)} regions detected"
        )

        return regions

    def generate_overall_mask(
        self,
        per_frame_scores: List[Dict[str, Any]],
        frames: Optional[List[np.ndarray]] = None,
        heatmaps: Optional[List[np.ndarray]] = None,
    ) -> Optional[np.ndarray]:
        """
        Generate overall manipulation mask across all frames.

        Args:
            per_frame_scores: List of per-frame scores
            frames: Optional list of frames for visualization
            heatmaps: Optional list of heatmaps

        Returns:
            Overall manipulation mask
        """
        if not per_frame_scores:
            return None

        if not frames and not heatmaps:
            scores = [f.get("manipulation_score", 0.0) for f in per_frame_scores]
            return np.array(scores)

        if heatmaps:
            if all(h is not None for h in heatmaps):
                heatmap_stack = np.stack(heatmaps, axis=0)
                return np.mean(heatmap_stack, axis=0)

        if frames:
            if len(frames) == len(per_frame_scores):
                overall_mask = np.zeros((len(frames),), dtype=np.float32)
                for i, frame in enumerate(frames):
                    if frame.ndim == 3:
                        overall_mask[i] = per_frame_scores[i].get("manipulation_score", 0.0)
                return overall_mask

        return None

    def localize(
        self,
        per_frame_scores: List[Dict[str, Any]],
        heatmaps: Optional[List[np.ndarray]] = None,
        face_bboxes: Optional[List[Optional[List[int]]]] = None,
        original_size: Tuple[int, int] = (640, 360),
        fps: int = 5,
    ) -> LocalizationResult:
        """
        Complete localization analysis.

        Args:
            per_frame_scores: Per-frame analysis results
            heatmaps: Optional list of per-frame heatmaps
            face_bboxes: Optional list of face bounding boxes
            original_size: Original frame size
            fps: Frames per second

        Returns:
            LocalizationResult with all localization data
        """
        temporal_regions, frame_timestamps = self.localize_temporal(
            per_frame_scores, fps
        )

        spatial_regions = []
        if heatmaps:
            for i, heatmap in enumerate(heatmaps):
                if heatmap is not None:
                    face_bbox = (
                        face_bboxes[i] if face_bboxes else None
                    )
                    frame_regions = self.localize_spatial(
                        heatmap, original_size, face_bbox
                    )
                    for region in frame_regions:
                        region.mask = cv2.resize(
                            region.mask, (original_size[0] // 4, original_size[1] // 4)
                        )
                    spatial_regions.extend(frame_regions)

        overall_mask = self.generate_overall_mask(per_frame_scores, heatmaps=heatmaps)

        overall_confidence = 0.0
        if temporal_regions:
            overall_confidence = np.mean([r.confidence for r in temporal_regions])
        elif per_frame_scores:
            scores = [f.get("manipulation_score", 0.0) for f in per_frame_scores]
            overall_confidence = float(np.mean(scores))

        result = LocalizationResult(
            temporal_regions=temporal_regions,
            spatial_regions=spatial_regions[:10],
            frame_timestamps=frame_timestamps,
            overall_manipulation_mask=overall_mask,
            confidence=overall_confidence,
        )

        logger.info(
            f"Localization complete: {len(temporal_regions)} temporal, "
            f"{len(spatial_regions)} spatial regions"
        )

        return result

    def create_spatial_overlay(
        self,
        original_image: np.ndarray,
        regions: List[SpatialRegion],
        alpha: float = 0.4,
    ) -> np.ndarray:
        """
        Create visualization overlay with spatial regions.

        Args:
            original_image: Original image
            regions: List of spatial regions
            alpha: Overlay transparency

        Returns:
            Overlay image
        """
        overlay = original_image.copy()

        for region in regions[:5]:
            mask_colored = np.zeros_like(original_image, dtype=np.uint8)

            if region.mask.ndim == 2:
                mask_colored[region.mask > 0.5] = [0, 0, 255]

            overlay = cv2.addWeighted(
                mask_colored,
                alpha,
                overlay,
                1 - alpha,
                0,
            )

            x1, y1, x2, y2 = region.bbox
            cv2.rectangle(
                overlay,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2,
            )

            cx, cy = region.centroid
            cv2.circle(
                overlay,
                (cx, cy),
                5,
                (0, 255, 255),
                -1,
            )

        return overlay

    def export_temporal_regions(
        self,
        regions: List[TemporalRegion],
        fps: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Export temporal regions with timestamps.

        Args:
            regions: List of temporal regions
            fps: Frames per second

        Returns:
            List of region dictionaries with timestamps
        """
        export = []
        for region in regions:
            export.append({
                "start_time_seconds": round(region.start_frame / fps, 3),
                "end_time_seconds": round(region.end_frame / fps, 3),
                "start_frame": region.start_frame,
                "end_frame": region.end_frame,
                "peak_frame": region.peak_frame,
                "duration_seconds": round(region.duration_frames / fps, 3),
                "confidence": region.confidence,
            })
        return export


def create_localizer() -> Localizer:
    """Factory function for Localizer."""
    config = get_config()
    return Localizer(
        peak_prominence=0.3,
        peak_distance=5,
        smoothing_window=11,
        mask_threshold=0.5,
        min_region_area=100,
        use_gpu=False,
    )
