"""
Unit tests for face detection module.

Tests face detection, alignment, and validation.
"""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock

from src.preprocessing.face_detector import (
    FaceBox,
    FaceDetector,
    MTCNN,
    detect_faces,
    align_face,
)


class TestFaceBox:
    """Test FaceBox dataclass."""

    def test_facebox_creation(self):
        """Test FaceBox object creation."""
        bbox = (100.0, 100.0, 200.0, 200.0)
        landmarks = ((150.0, 120.0), (180.0, 120.0), (165.0, 150.0), (155.0, 170.0), (175.0, 170.0))

        face = FaceBox(
            bbox=bbox,
            landmarks=landmarks,
            confidence=0.95,
        )

        assert face.bbox == bbox
        assert face.landmarks == landmarks
        assert face.confidence == 0.95
        assert face.aligned_image is None

    def test_facebox_with_aligned_image(self):
        """Test FaceBox with aligned image."""
        face = FaceBox(
            bbox=(100, 100, 200, 200),
            landmarks=((150, 120), (180, 120), (165, 150), (155, 170), (175, 170)),
            confidence=0.9,
            aligned_image=np.zeros((224, 224, 3), dtype=np.uint8),
        )

        assert face.aligned_image is not None
        assert face.aligned_image.shape == (224, 224, 3)

    def test_facebox_to_dict(self):
        """Test FaceBox serialization to dictionary."""
        face = FaceBox(
            bbox=(100.0, 100.0, 200.0, 200.0),
            landmarks=((150.0, 120.0), (180.0, 120.0), (165.0, 150.0), (155.0, 170.0), (175.0, 170.0)),
            confidence=0.95,
        )

        face_dict = face.to_dict()

        assert face_dict["bbox"] == [100.0, 100.0, 200.0, 200.0]
        assert len(face_dict["landmarks"]) == 5
        assert face_dict["confidence"] == 0.95


class TestMTCNN:
    """Test MTCNN face detector."""

    def test_mtcnn_initialization(self):
        """Test MTCNN initialization."""
        detector = MTCNN()

        assert detector.device is not None
        assert detector.p_thresh == 0.6
        assert detector.r_thresh == 0.7
        assert detector.o_thresh == 0.8
        assert detector.min_face_size == 20

    def test_mtcnn_with_custom_thresholds(self):
        """Test MTCNN with custom thresholds."""
        detector = MTCNN(
            p_thresh=0.5,
            r_thresh=0.6,
            o_thresh=0.7,
            min_face_size=30,
        )

        assert detector.p_thresh == 0.5
        assert detector.r_thresh == 0.6
        assert detector.o_thresh == 0.7
        assert detector.min_face_size == 30

    def test_validate_image_none(self):
        """Test validation of None image."""
        detector = MTCNN()

        is_valid, error_msg = detector.validate_image(None)

        assert is_valid is False
        assert "None" in error_msg

    def test_validate_image_wrong_type(self):
        """Test validation of wrong image type."""
        detector = MTCNN()

        is_valid, error_msg = detector.validate_image("not an image")

        assert is_valid is False
        assert "numpy array" in error_msg

    def test_validate_image_wrong_dimensions(self):
        """Test validation of wrong image dimensions."""
        detector = MTCNN()

        # 2D array instead of 3D
        is_valid, error_msg = detector.validate_image(np.zeros((100, 100)))

        assert is_valid is False
        assert "3D" in error_msg

    def test_validate_image_too_small(self):
        """Test validation of too small image."""
        detector = MTCNN()

        is_valid, error_msg = detector.validate_image(np.zeros((50, 50, 3)))

        assert is_valid is False
        assert "too small" in error_msg.lower()

    def test_validate_image_valid(self):
        """Test validation of valid image."""
        detector = MTCNN()

        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        is_valid, error_msg = detector.validate_image(image)

        assert is_valid is True
        assert error_msg == ""

    def test_validate_image_4_channels(self):
        """Test validation of 4-channel image."""
        detector = MTCNN()

        image = np.random.randint(0, 255, (100, 100, 4), dtype=np.uint8)
        is_valid, error_msg = detector.validate_image(image)

        assert is_valid is True

    def test_detect_empty_image(self):
        """Test detection on empty image."""
        detector = MTCNN()

        faces = detector.detect(None)
        assert faces == []

        faces = detector.detect(np.array([]))
        assert faces == []


class TestFaceDetector:
    """Test high-level FaceDetector class."""

    def test_detector_initialization(self):
        """Test FaceDetector initialization."""
        detector = FaceDetector()

        assert detector.device is not None
        assert detector.batch_size == 8
        assert detector.mtcnn is not None

    def test_detector_with_custom_params(self):
        """Test FaceDetector with custom parameters."""
        detector = FaceDetector(
            batch_size=16,
            min_face_size=40,
        )

        assert detector.batch_size == 16
        assert detector.mtcnn.min_face_size == 40

    def test_detect_faces_empty_image(self):
        """Test face detection on empty image."""
        detector = FaceDetector()

        faces = detector.detect_faces(np.array([]))
        assert faces == []

    def test_detect_faces_small_image(self):
        """Test face detection on small image."""
        detector = FaceDetector()

        image = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
        faces = detector.detect_faces(image)

        assert faces == []

    def test_detect_faces_batch_empty(self):
        """Test batch detection with empty list."""
        detector = FaceDetector()

        results = detector.detect_faces_batch([])

        assert results == []

    def test_get_largest_face_no_faces(self):
        """Test getting largest face when no faces detected."""
        detector = FaceDetector()

        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        largest = detector.get_largest_face(image)

        assert largest is None


class TestConvenienceFunctions:
    """Test convenience functions."""

    def test_detect_faces_function(self):
        """Test the detect_faces convenience function."""
        # Mock the FaceDetector class
        with patch('src.preprocessing.face_detector.FaceDetector') as MockDetector:
            mock_instance = MagicMock()
            mock_instance.detect_faces.return_value = []
            MockDetector.return_value = mock_instance

            from src.preprocessing import detect_faces

            image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
            detect_faces(image)

            mock_instance.detect_faces.assert_called_once()

    def test_align_face_function(self):
        """Test the align_face convenience function."""
        face = FaceBox(
            bbox=(100, 100, 200, 200),
            landmarks=((150, 120), (180, 120), (165, 150), (155, 170), (175, 170)),
            confidence=0.95,
        )

        # Mock the alignment
        with patch('src.preprocessing.face_detector.FaceDetector') as MockDetector:
            mock_mtcnn = MagicMock()
            mock_mtcnn.align_face.return_value = np.zeros((224, 224, 3), dtype=np.uint8)

            mock_instance = MagicMock()
            mock_instance.mtcnn = mock_mtcnn
            MockDetector.return_value = mock_instance

            from src.preprocessing import align_face

            image = np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8)
            result = align_face(image, face)

            mock_mtcnn.align_face.assert_called_once()
            assert result is not None


class TestFaceDetectorIntegration:
    """Integration tests for FaceDetector."""

    def test_detect_faces_with_mock(self, tmp_path):
        """Test face detection with mocked MTCNN."""
        # Create a mock image
        image = np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8)

        # Mock the MTCNN to return a face
        mock_face = FaceBox(
            bbox=(100.0, 100.0, 200.0, 200.0),
            landmarks=((150.0, 120.0), (180.0, 120.0), (165.0, 150.0), (155.0, 170.0), (175.0, 170.0)),
            confidence=0.95,
        )

        with patch.object(MTCNN, 'detect', return_value=[mock_face]):
            detector = FaceDetector()
            faces = detector.detect_faces(image)

            assert len(faces) == 1
            assert faces[0].confidence == 0.95

    def test_detect_faces_batch_with_mock(self, tmp_path):
        """Test batch face detection with mocked MTCNN."""
        images = [
            np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8)
            for _ in range(3)
        ]

        mock_face = FaceBox(
            bbox=(100, 100, 200, 200),
            landmarks=((150, 120), (180, 120), (165, 150), (155, 170), (175, 170)),
            confidence=0.9,
        )

        with patch.object(MTCNN, 'detect_batch', return_value=[[mock_face] for _ in images]):
            detector = FaceDetector()
            results = detector.detect_faces_batch(images)

            assert len(results) == 3
            for faces in results:
                assert len(faces) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
