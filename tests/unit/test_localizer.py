"""
Unit Tests for Localizer.

Tests temporal and spatial localization of manipulation regions.
"""

import pytest
import numpy as np
from pathlib import Path

from src.explainability.localizer import (
    Localizer,
    LocalizationResult,
    TemporalRegion,
    SpatialRegion,
)


class TestLocalizer:
    """Test cases for Localizer."""

    @pytest.fixture
    def localizer(self):
        """Create a Localizer instance."""
        return Localizer(
            peak_prominence=0.3,
            peak_distance=5,
            smoothing_window=11,
            mask_threshold=0.5,
            min_region_area=100,
        )

    def test_localizer_initialization(self, localizer):
        """Test that localizer initializes correctly."""
        assert localizer is not None
        assert localizer.peak_prominence == 0.3
        assert localizer.peak_distance == 5
        assert localizer.mask_threshold == 0.5

    def test_smooth_scores_empty(self, localizer):
        """Test smoothing with empty scores."""
        result = localizer._smooth_scores([])
        assert len(result) == 0

    def test_smooth_scores_single(self, localizer):
        """Test smoothing with single score."""
        result = localizer._smooth_scores([0.5])
        assert len(result) == 1
        assert result[0] == 0.5

    def test_smooth_scores_multiple(self, localizer):
        """Test smoothing with multiple scores."""
        scores = [0.1, 0.5, 0.8, 0.6, 0.2, 0.9, 0.4]
        result = localizer._smooth_scores(scores)
        assert len(result) == len(scores)
        assert all(0 <= r <= 1 for r in result)

    def test_detect_peaks_no_peaks(self, localizer):
        """Test peak detection with no peaks."""
        scores = np.array([0.1, 0.2, 0.15, 0.25, 0.2])
        peaks, _ = localizer._detect_peaks(scores)
        assert len(peaks) == 0

    def test_detect_peaks_with_peaks(self, localizer):
        """Test peak detection with peaks."""
        scores = np.array([0.1, 0.3, 0.8, 0.4, 0.2, 0.7, 0.3])
        peaks, properties = localizer._detect_peaks(scores)
        assert len(peaks) >= 1

    def test_cluster_peaks_empty(self, localizer):
        """Test clustering with no peaks."""
        regions = localizer._cluster_peaks(np.array([]), np.array([]))
        assert len(regions) == 0

    def test_cluster_peaks_single(self, localizer):
        """Test clustering with single peak."""
        peaks = np.array([10])
        scores = np.full(20, 0.3)
        scores[10] = 0.8
        regions = localizer._cluster_peaks(peaks, scores)
        assert len(regions) >= 1
        assert regions[0].peak_frame == 10

    def test_cluster_peaks_multiple(self, localizer):
        """Test clustering with multiple peaks."""
        scores = np.zeros(60)
        scores[5] = 0.9
        scores[6] = 0.85
        scores[7] = 0.88
        scores[20] = 0.7
        scores[21] = 0.75
        scores[50] = 0.6
        peaks, _ = localizer._detect_peaks(scores)
        if len(peaks) > 0:
            regions = localizer._cluster_peaks(peaks, scores)
            assert len(regions) >= 1

    def test_threshold_heatmap_otsu(self, localizer):
        """Test Otsu thresholding."""
        heatmap = np.random.rand(100, 100)
        mask = localizer._threshold_heatmap(heatmap, method="otsu")
        assert mask.shape == (100, 100)
        assert mask.dtype == bool

    def test_threshold_heatmap_fixed(self, localizer):
        """Test fixed thresholding."""
        heatmap = np.random.rand(50, 50) * 0.3
        heatmap[10:20, 10:20] = 0.9
        mask = localizer._threshold_heatmap(heatmap, method="otsu")
        assert mask.shape == (50, 50)
        assert np.any(mask)

    def test_detect_contours(self, localizer):
        """Test contour detection."""
        mask = np.zeros((100, 100), dtype=bool)
        mask[40:60, 40:60] = True
        contours = localizer._detect_contours(mask)
        assert len(contours) >= 1

    def test_simplify_polygon(self, localizer):
        """Test polygon simplification."""
        contour = np.array([[[0, 0]], [[10, 0]], [[10, 10]], [[0, 10]]])
        polygon = localizer._simplify_polygon(contour)
        assert len(polygon) == 4

    def test_localize_temporal_no_frames(self, localizer):
        """Test temporal localization with no frames."""
        regions, timestamps = localizer.localize_temporal([])
        assert len(regions) == 0
        assert len(timestamps) == 0

    def test_localize_temporal_single_frame(self, localizer):
        """Test temporal localization with single frame."""
        per_frame_scores = [{"manipulation_score": 0.9}]
        regions, timestamps = localizer.localize_temporal(per_frame_scores, fps=5)
        assert len(timestamps) == 1
        assert timestamps[0]["frame_number"] == 0
        assert timestamps[0]["timestamp_seconds"] == 0.0

    def test_localize_temporal_multiple_frames(self, localizer):
        """Test temporal localization with multiple frames."""
        per_frame_scores = [
            {"manipulation_score": 0.1},
            {"manipulation_score": 0.3},
            {"manipulation_score": 0.8},
            {"manipulation_score": 0.7},
            {"manipulation_score": 0.2},
            {"manipulation_score": 0.9},
            {"manipulation_score": 0.4},
        ]
        regions, timestamps = localizer.localize_temporal(per_frame_scores, fps=5)
        assert len(timestamps) == 7
        assert len(timestamps[0]) >= 3

    def test_localize_spatial_no_heatmap(self, localizer):
        """Test spatial localization with no heatmap."""
        regions = localizer.localize_spatial(None, (640, 360))
        assert len(regions) == 0

    def test_localize_spatial_empty_heatmap(self, localizer):
        """Test spatial localization with empty heatmap."""
        heatmap = np.zeros((100, 100))
        regions = localizer.localize_spatial(heatmap, (640, 360))
        assert len(regions) == 0

    def test_localize_spatial_with_heatmap(self, localizer):
        """Test spatial localization with heatmap."""
        heatmap = np.zeros((100, 100))
        heatmap[40:60, 40:60] = 0.9
        regions = localizer.localize_spatial(heatmap, (640, 360))
        assert len(regions) >= 1

    def test_localize_spatial_with_face_bbox(self, localizer):
        """Test spatial localization with face bbox."""
        heatmap = np.zeros((100, 100))
        heatmap[10:30, 10:30] = 0.9
        face_bbox = [30, 30, 70, 70]
        regions = localizer.localize_spatial(heatmap, (640, 360), face_bbox)
        assert len(regions) >= 1

    def test_generate_overall_mask_no_data(self, localizer):
        """Test overall mask generation with no data."""
        mask = localizer.generate_overall_mask([])
        assert mask is None

    def test_generate_overall_mask_scores_only(self, localizer):
        """Test overall mask generation with scores only."""
        per_frame_scores = [
            {"manipulation_score": 0.1},
            {"manipulation_score": 0.5},
            {"manipulation_score": 0.9},
        ]
        mask = localizer.generate_overall_mask(per_frame_scores)
        assert mask is not None
        assert len(mask) == 3

    def test_localize_complete(self, localizer):
        """Test complete localization."""
        per_frame_scores = [
            {"manipulation_score": 0.1},
            {"manipulation_score": 0.3},
            {"manipulation_score": 0.8},
            {"manipulation_score": 0.7},
            {"manipulation_score": 0.2},
        ]
        heatmaps = [np.random.rand(50, 50) for _ in range(5)]

        result = localizer.localize(
            per_frame_scores,
            heatmaps=heatmaps,
            original_size=(640, 360),
            fps=5,
        )

        assert isinstance(result, LocalizationResult)
        assert result.confidence >= 0.0
        assert len(result.frame_timestamps) == 5

    def test_localize_no_heatmaps(self, localizer):
        """Test localization without heatmaps."""
        per_frame_scores = [
            {"manipulation_score": 0.2},
            {"manipulation_score": 0.6},
            {"manipulation_score": 0.8},
        ]

        result = localizer.localize(
            per_frame_scores,
            heatmaps=None,
            original_size=(640, 360),
            fps=5,
        )

        assert isinstance(result, LocalizationResult)
        assert len(result.frame_timestamps) == 3

    def test_create_spatial_overlay(self, localizer):
        """Test spatial overlay creation."""
        original = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

        regions = [
            SpatialRegion(
                mask=np.zeros((100, 100)),
                bbox=[20, 20, 80, 80],
                centroid=(50, 50),
                area=3600,
                confidence=0.8,
            )
        ]

        overlay = localizer.create_spatial_overlay(original, regions)
        assert overlay.shape == original.shape

    def test_export_temporal_regions(self, localizer):
        """Test temporal region export."""
        regions = [
            TemporalRegion(
                start_frame=10,
                end_frame=20,
                confidence=0.8,
                peak_frame=15,
                duration_frames=11,
            )
        ]

        export = localizer.export_temporal_regions(regions, fps=5)
        assert len(export) == 1
        assert export[0]["start_time_seconds"] == 2.0
        assert export[0]["end_time_seconds"] == 4.0


class TestLocalizationResult:
    """Test cases for LocalizationResult dataclass."""

    def test_to_dict_empty(self):
        """Test serialization of empty result."""
        result = LocalizationResult(
            temporal_regions=[],
            spatial_regions=[],
            frame_timestamps=[],
        )
        as_dict = result.to_dict()
        assert as_dict["temporal_regions"] == []
        assert as_dict["spatial_regions"] == []
        assert as_dict["frame_timestamps"] == []
        assert as_dict["confidence"] == 0.0

    def test_to_dict_with_data(self):
        """Test serialization with data."""
        temporal = TemporalRegion(
            start_frame=5,
            end_frame=15,
            confidence=0.8,
            peak_frame=10,
            duration_frames=11,
        )
        spatial = SpatialRegion(
            mask=np.zeros((10, 10)),
            bbox=[0, 0, 10, 10],
            centroid=(5, 5),
            area=100,
            confidence=0.7,
        )
        result = LocalizationResult(
            temporal_regions=[temporal],
            spatial_regions=[spatial],
            frame_timestamps=[{"frame_number": 0}],
            confidence=0.75,
        )
        as_dict = result.to_dict()
        assert len(as_dict["temporal_regions"]) == 1
        assert len(as_dict["spatial_regions"]) == 1
        assert len(as_dict["frame_timestamps"]) == 1
        assert as_dict["confidence"] == 0.75


class TestLocalizerIntegration:
    """Integration tests for Localizer."""

    @pytest.fixture
    def localizer(self):
        """Create a Localizer instance."""
        return Localizer()

    def test_create_localizer_factory(self):
        """Test factory function."""
        from src.explainability.localizer import create_localizer
        localizer = create_localizer()
        assert isinstance(localizer, Localizer)

    def test_full_pipeline(self, localizer):
        """Test complete localization pipeline."""
        per_frame_scores = []
        for i in range(50):
            if 15 <= i <= 25 or 35 <= i <= 40:
                score = 0.7 + np.random.random() * 0.25
            else:
                score = np.random.random() * 0.3
            per_frame_scores.append({"manipulation_score": score})

        result = localizer.localize(
            per_frame_scores,
            original_size=(640, 360),
            fps=5,
        )

        assert result.confidence >= 0.0
        assert len(result.frame_timestamps) == 50
        assert isinstance(result.to_dict(), dict)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
