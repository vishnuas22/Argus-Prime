"""
Unit tests for face alignment module.

Tests similarity transform, histogram equalization, and alignment pipeline.
"""

import pytest
import numpy as np
import cv2
from unittest.mock import patch, MagicMock

from src.preprocessing.face_aligner import (
    FaceAligner,
    Landmark5Points,
    AlignedFace,
    align_face,
    align_face_from_array,
)


class TestLandmark5Points:
    """Test Landmark5Points dataclass."""

    def test_landmark_creation(self):
        """Test landmark creation."""
        landmarks = Landmark5Points(
            left_eye=(100.0, 50.0),
            right_eye=(150.0, 50.0),
            nose=(125.0, 80.0),
            left_mouth=(110.0, 120.0),
            right_mouth=(140.0, 120.0),
        )

        assert landmarks.left_eye == (100.0, 50.0)
        assert landmarks.right_eye == (150.0, 50.0)
        assert landmarks.nose == (125.0, 80.0)

    def test_to_array(self):
        """Test conversion to array."""
        landmarks = Landmark5Points(
            left_eye=(100.0, 50.0),
            right_eye=(150.0, 50.0),
            nose=(125.0, 80.0),
            left_mouth=(110.0, 120.0),
            right_mouth=(140.0, 120.0),
        )

        arr = landmarks.to_array()

        assert arr.shape == (5, 2)
        assert np.array_equal(arr[0], [100.0, 50.0])
        assert np.array_equal(arr[2], [125.0, 80.0])

    def test_from_array(self):
        """Test creation from array."""
        arr = np.array([
            [100.0, 50.0],
            [150.0, 50.0],
            [125.0, 80.0],
            [110.0, 120.0],
            [140.0, 120.0],
        ])

        landmarks = Landmark5Points.from_array(arr)

        assert landmarks.left_eye == (100.0, 50.0)
        assert landmarks.right_eye == (150.0, 50.0)
        assert landmarks.nose == (125.0, 80.0)
        assert landmarks.left_mouth == (110.0, 120.0)
        assert landmarks.right_mouth == (140.0, 120.0)


class TestAlignedFace:
    """Test AlignedFace dataclass."""

    def test_aligned_face_creation(self):
        """Test aligned face creation."""
        image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        landmarks = Landmark5Points(
            left_eye=(100.0, 50.0),
            right_eye=(150.0, 50.0),
            nose=(125.0, 80.0),
            left_mouth=(110.0, 120.0),
            right_mouth=(140.0, 120.0),
        )
        matrix = np.eye(2, 3)

        aligned = AlignedFace(
            image=image,
            landmarks=landmarks,
            transformation_matrix=matrix,
            crop_size=(224, 224),
        )

        assert aligned.image.shape == (224, 224, 3)
        assert aligned.crop_size == (224, 224)

    def test_to_rgb(self):
        """Test BGR to RGB conversion."""
        bgr_image = np.zeros((100, 100, 3), dtype=np.uint8)
        bgr_image[:, :, 2] = 255

        aligned = AlignedFace(
            image=bgr_image,
            landmarks=MagicMock(),
            transformation_matrix=np.eye(2, 3),
            crop_size=(100, 100),
        )

        rgb = aligned.to_rgb()

        assert rgb[0, 0, 0] == 255
        assert rgb[0, 0, 2] == 0


class TestFaceAligner:
    """Test FaceAligner class."""

    def test_aligner_initialization(self):
        """Test aligner initialization."""
        aligner = FaceAligner()

        assert aligner.output_size == (224, 224)
        assert aligner.enable_histogram_equalization is True
        assert aligner.reference_landmarks.shape == (5, 2)

    def test_aligner_custom_params(self):
        """Test aligner with custom parameters."""
        aligner = FaceAligner(
            output_size=(256, 256),
            enable_histogram_equalization=False,
        )

        assert aligner.output_size == (256, 256)
        assert aligner.enable_histogram_equalization is False

    def test_validate_landmarks_valid(self):
        """Test landmark validation with valid landmarks."""
        aligner = FaceAligner()
        landmarks = Landmark5Points(
            left_eye=(100.0, 50.0),
            right_eye=(150.0, 50.0),
            nose=(125.0, 80.0),
            left_mouth=(110.0, 120.0),
            right_mouth=(140.0, 120.0),
        )

        assert aligner._validate_landmarks(landmarks) is True

    def test_validate_landmarks_nan(self):
        """Test landmark validation with NaN values."""
        aligner = FaceAligner()
        landmarks = Landmark5Points(
            left_eye=(np.nan, 50.0),
            right_eye=(150.0, 50.0),
            nose=(125.0, 80.0),
            left_mouth=(110.0, 120.0),
            right_mouth=(140.0, 120.0),
        )

        assert aligner._validate_landmarks(landmarks) is False

    def test_validate_landmarks_inf(self):
        """Test landmark validation with Inf values."""
        aligner = FaceAligner()
        landmarks = Landmark5Points(
            left_eye=(np.inf, 50.0),
            right_eye=(150.0, 50.0),
            nose=(125.0, 80.0),
            left_mouth=(110.0, 120.0),
            right_mouth=(140.0, 120.0),
        )

        assert aligner._validate_landmarks(landmarks) is False

    def test_validate_landmarks_too_close(self):
        """Test landmark validation with eyes too close."""
        aligner = FaceAligner()
        landmarks = Landmark5Points(
            left_eye=(100.0, 50.0),
            right_eye=(101.0, 50.0),
            nose=(125.0, 80.0),
            left_mouth=(110.0, 120.0),
            right_mouth=(140.0, 120.0),
        )

        assert aligner._validate_landmarks(landmarks) is False

    def test_compute_similarity_transform_identity(self):
        """Test similarity transform with identical points."""
        aligner = FaceAligner()
        source = np.array([
            [0.3573, 0.4752],
            [0.6427, 0.4752],
            [0.5000, 0.7382],
            [0.3827, 0.8415],
            [0.6173, 0.8415],
        ], dtype=np.float64)
        target = source.copy()

        matrix = aligner._compute_similarity_transform(source, target)

        assert matrix.shape == (2, 3)
        np.testing.assert_array_almost_equal(matrix[:2, :2], np.eye(2), decimal=5)

    def test_compute_similarity_transform_scale(self):
        """Test similarity transform with scaled points."""
        aligner = FaceAligner()
        source = np.array([
            [0.3573, 0.4752],
            [0.6427, 0.4752],
            [0.5000, 0.7382],
            [0.3827, 0.8415],
            [0.6173, 0.8415],
        ], dtype=np.float64)
        target = source * 2

        matrix = aligner._compute_similarity_transform(source, target)

        assert matrix.shape == (2, 3)

    def test_warp_image(self):
        """Test image warping."""
        aligner = FaceAligner()
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        matrix = np.eye(2, 3)
        matrix[0, 2] = 10

        warped = aligner._warp_image(image, matrix, (100, 100))

        assert warped.shape == (100, 100, 3)

    def test_apply_histogram_equalization(self):
        """Test histogram equalization."""
        aligner = FaceAligner()
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

        equalized = aligner._apply_histogram_equalization(image)

        assert equalized.shape == image.shape
        assert equalized.dtype == np.uint8

    def test_apply_histogram_equalization_grayscale(self):
        """Test histogram equalization with grayscale."""
        aligner = FaceAligner()
        image = np.random.randint(0, 255, (100, 100), dtype=np.uint8)

        equalized = aligner._apply_histogram_equalization(image)

        assert equalized.shape == image.shape

    def test_align(self):
        """Test face alignment."""
        aligner = FaceAligner()
        face_image = np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8)
        landmarks = Landmark5Points(
            left_eye=(100.0, 150.0),
            right_eye=(200.0, 150.0),
            nose=(150.0, 200.0),
            left_mouth=(120.0, 250.0),
            right_mouth=(180.0, 250.0),
        )

        result = aligner.align(face_image, landmarks)

        assert isinstance(result, AlignedFace)
        assert result.image.shape[:2] == (224, 224)
        assert result.crop_size == (224, 224)
        assert result.transformation_matrix.shape == (2, 3)

    def test_align_with_custom_size(self):
        """Test alignment with custom output size."""
        aligner = FaceAligner(output_size=(256, 256))
        face_image = np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8)
        landmarks = Landmark5Points(
            left_eye=(100.0, 150.0),
            right_eye=(200.0, 150.0),
            nose=(150.0, 200.0),
            left_mouth=(120.0, 250.0),
            right_mouth=(180.0, 250.0),
        )

        result = aligner.align(face_image, landmarks)

        assert result.image.shape[:2] == (256, 256)
        assert result.crop_size == (256, 256)

    def test_align_without_histogram_equalization(self):
        """Test alignment without histogram equalization."""
        aligner = FaceAligner(enable_histogram_equalization=False)
        face_image = np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8)
        landmarks = Landmark5Points(
            left_eye=(100.0, 150.0),
            right_eye=(200.0, 150.0),
            nose=(150.0, 200.0),
            left_mouth=(120.0, 250.0),
            right_mouth=(180.0, 250.0),
        )

        result = aligner.align(face_image, landmarks)

        assert result.image.shape[:2] == (224, 224)

    def test_align_invalid_landmarks(self):
        """Test alignment with invalid landmarks."""
        aligner = FaceAligner()
        face_image = np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8)
        landmarks = Landmark5Points(
            left_eye=(np.nan, 150.0),
            right_eye=(200.0, 150.0),
            nose=(150.0, 200.0),
            left_mouth=(120.0, 250.0),
            right_mouth=(180.0, 250.0),
        )

        with pytest.raises(ValueError, match="Invalid landmarks"):
            aligner.align(face_image, landmarks)

    def test_align_empty_image(self):
        """Test alignment with empty image."""
        aligner = FaceAligner()
        face_image = np.array([], dtype=np.uint8)
        landmarks = Landmark5Points(
            left_eye=(100.0, 150.0),
            right_eye=(200.0, 150.0),
            nose=(150.0, 200.0),
            left_mouth=(120.0, 250.0),
            right_mouth=(180.0, 250.0),
        )

        with pytest.raises(ValueError, match="Empty face image"):
            aligner.align(face_image, landmarks)

    def test_align_from_array(self):
        """Test alignment from landmarks array."""
        aligner = FaceAligner()
        face_image = np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8)
        landmarks_array = np.array([
            [100.0, 150.0],
            [200.0, 150.0],
            [150.0, 200.0],
            [120.0, 250.0],
            [180.0, 250.0],
        ])

        result = aligner.align_from_array(face_image, landmarks_array)

        assert isinstance(result, AlignedFace)
        assert result.image.shape[:2] == (224, 224)

    def test_create_reference_template(self):
        """Test reference template creation."""
        aligner = FaceAligner()
        template = aligner.create_reference_template((224, 224))

        assert template.shape == (224, 224, 3)
        assert template.dtype == np.uint8


class TestConvenienceFunctions:
    """Test convenience functions."""

    def test_align_face_function(self):
        """Test align_face convenience function."""
        face_image = np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8)
        landmarks = Landmark5Points(
            left_eye=(100.0, 150.0),
            right_eye=(200.0, 150.0),
            nose=(150.0, 200.0),
            left_mouth=(120.0, 250.0),
            right_mouth=(180.0, 250.0),
        )

        result = align_face(face_image, landmarks)

        assert isinstance(result, AlignedFace)
        assert result.image.shape[:2] == (224, 224)

    def test_align_face_from_array_function(self):
        """Test align_face_from_array convenience function."""
        face_image = np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8)
        landmarks_array = np.array([
            [100.0, 150.0],
            [200.0, 150.0],
            [150.0, 200.0],
            [120.0, 250.0],
            [180.0, 250.0],
        ])

        result = align_face_from_array(face_image, landmarks_array)

        assert isinstance(result, AlignedFace)
        assert result.image.shape[:2] == (224, 224)

    def test_align_face_custom_size(self):
        """Test align_face with custom size."""
        face_image = np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8)
        landmarks = Landmark5Points(
            left_eye=(100.0, 150.0),
            right_eye=(200.0, 150.0),
            nose=(150.0, 200.0),
            left_mouth=(120.0, 250.0),
            right_mouth=(180.0, 250.0),
        )

        result = align_face(
            face_image,
            landmarks,
            output_size=(256, 256),
            enable_histogram_equalization=False,
        )

        assert result.image.shape[:2] == (256, 256)


class TestFaceAlignerIntegration:
    """Integration tests for FaceAligner."""

    def test_align_realistic_face(self):
        """Test alignment with realistic face proportions."""
        aligner = FaceAligner()

        face_image = np.random.randint(0, 255, (500, 500, 3), dtype=np.uint8)

        left_eye = (175.0, 200.0)
        right_eye = (325.0, 200.0)
        nose = (250.0, 300.0)
        left_mouth = (200.0, 400.0)
        right_mouth = (300.0, 400.0)

        landmarks = Landmark5Points(
            left_eye=left_eye,
            right_eye=right_eye,
            nose=nose,
            left_mouth=left_mouth,
            right_mouth=right_mouth,
        )

        result = aligner.align(face_image, landmarks)

        assert result.image.shape == (224, 224, 3)
        assert result.transformation_matrix.shape == (2, 3)

    def test_align_multiple_times(self):
        """Test that aligner can be reused."""
        aligner = FaceAligner()

        for _ in range(3):
            face_image = np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8)
            landmarks = Landmark5Points(
                left_eye=(100.0, 150.0),
                right_eye=(200.0, 150.0),
                nose=(150.0, 200.0),
                left_mouth=(120.0, 250.0),
                right_mouth=(180.0, 250.0),
            )

            result = aligner.align(face_image, landmarks)
            assert result.image.shape[:2] == (224, 224)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
