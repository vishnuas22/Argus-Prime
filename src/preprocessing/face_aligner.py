"""
Face alignment using 5-point landmarks.

Implements similarity transform to align faces to a reference template.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np
from scipy.spatial import distance

from src.config import get_config
from src.core.logging import logger


@dataclass
class Landmark5Points:
    """5-point facial landmarks."""
    left_eye: Tuple[float, float]
    right_eye: Tuple[float, float]
    nose: Tuple[float, float]
    left_mouth: Tuple[float, float]
    right_mouth: Tuple[float, float]

    def to_array(self) -> np.ndarray:
        """Convert to numpy array for transformation."""
        return np.array([
            self.left_eye,
            self.right_eye,
            self.nose,
            self.left_mouth,
            self.right_mouth,
        ], dtype=np.float64)

    @classmethod
    def from_array(cls, arr: np.ndarray) -> "Landmark5Points":
        """Create from numpy array."""
        return cls(
            left_eye=tuple(arr[0]),
            right_eye=tuple(arr[1]),
            nose=tuple(arr[2]),
            left_mouth=tuple(arr[3]),
            right_mouth=tuple(arr[4]),
        )


@dataclass
class AlignedFace:
    """Result of face alignment."""
    image: np.ndarray
    landmarks: Landmark5Points
    transformation_matrix: np.ndarray
    crop_size: Tuple[int, int]

    def to_rgb(self) -> np.ndarray:
        """Convert BGR to RGB."""
        return cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)


class FaceAligner:
    """
    Face alignment using 5-point landmarks.

    Aligns faces to a reference template using similarity transform.

    Reference landmarks based on FFHQ/CelebA dataset alignment:
    - Left eye: (35.73, 47.52) % of crop
    - Right eye: (64.27, 47.52) % of crop
    - Nose: (50.00, 73.82) % of crop
    - Left mouth: (38.27, 84.15) % of crop
    - Right mouth: (61.73, 84.15) % of crop

    Attributes:
        output_size: Output face size (width, height)
        enable_histogram_equalization: Whether to apply histogram equalization
    """

    DEFAULT_OUTPUT_SIZE = (224, 224)

    DEFAULT_REFERENCE_LANDMARKS = np.array([
        [0.3573, 0.4752],
        [0.6427, 0.4752],
        [0.5000, 0.7382],
        [0.3827, 0.8415],
        [0.6173, 0.8415],
    ], dtype=np.float64)

    def __init__(
        self,
        output_size: Tuple[int, int] = (224, 224),
        enable_histogram_equalization: bool = True,
        reference_landmarks: Optional[np.ndarray] = None,
    ):
        """
        Initialize the face aligner.

        Args:
            output_size: Output face size (width, height)
            enable_histogram_equalization: Whether to apply histogram equalization
            reference_landmarks: 5-point reference landmarks (normalized 0-1)
        """
        self.output_size = output_size
        self.enable_histogram_equalization = enable_histogram_equalization
        self.reference_landmarks = (
            reference_landmarks if reference_landmarks is not None
            else self.DEFAULT_REFERENCE_LANDMARKS
        )

        logger.info(
            f"FaceAligner initialized: size={output_size}, "
            f"hist_eq={enable_histogram_equalization}"
        )

    def _compute_similarity_transform(
        self,
        source_points: np.ndarray,
        target_points: np.ndarray,
    ) -> np.ndarray:
        """
        Compute similarity transform between two point sets.

        Args:
            source_points: Source landmarks (5, 2)
            target_points: Target landmarks (5, 2)

        Returns:
            2x3 similarity transformation matrix
        """
        assert source_points.shape == target_points.shape
        assert source_points.shape == (5, 2)

        source_centroid = np.mean(source_points, axis=0)
        target_centroid = np.mean(target_points, axis=0)

        source_centered = source_points - source_centroid
        target_centered = target_points - target_centroid

        source_norm = np.sqrt(np.sum(source_centered ** 2))
        target_norm = np.sqrt(np.sum(target_centered ** 2))

        scale = target_norm / source_norm if source_norm > 0 else 1.0

        covariance = source_centered.T @ target_centered
        u, s, vh = np.linalg.svd(covariance)

        reflection = np.eye(2)
        if np.linalg.det(u) * np.linalg.det(vh) < 0:
            reflection[1, 1] = -1

        rotation = u @ reflection @ vh

        matrix = np.eye(2, 3, dtype=np.float64)
        matrix[:2, :2] = scale * rotation
        matrix[0, 2] = target_centroid[0] - scale * rotation[0] @ source_centroid
        matrix[1, 2] = target_centroid[1] - scale * rotation[1] @ source_centroid

        return matrix

    def _warp_image(
        self,
        image: np.ndarray,
        transform_matrix: np.ndarray,
        output_size: Tuple[int, int],
    ) -> np.ndarray:
        """
        Warp image using similarity transform.

        Args:
            image: Input image (H, W, C)
            transform_matrix: 2x3 transformation matrix
            output_size: Output size (width, height)

        Returns:
            Warped image
        """
        return cv2.warpAffine(
            image,
            transform_matrix,
            output_size,
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )

    def _apply_histogram_equalization(self, image: np.ndarray) -> np.ndarray:
        """
        Apply histogram equalization to face image.

        Args:
            image: Input face image (BGR)

        Returns:
            Equalized image
        """
        if len(image.shape) != 3 or image.shape[2] != 3:
            return image

        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)

        lab = cv2.merge((l, a, b))
        equalized = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

        return equalized

    def _validate_landmarks(self, landmarks: Landmark5Points) -> bool:
        """Validate that landmarks are reasonable."""
        points = landmarks.to_array()

        if np.any(np.isnan(points)) or np.any(np.isinf(points)):
            return False

        if np.any(points < 0) or np.any(points > 10000):
            return False

        left_eye = np.array(landmarks.left_eye)
        right_eye = np.array(landmarks.right_eye)
        eye_distance = distance.euclidean(left_eye, right_eye)

        if eye_distance < 5:
            return False

        return True

    def align(
        self,
        face_image: np.ndarray,
        landmarks: Landmark5Points,
        validate_input: bool = True,
    ) -> AlignedFace:
        """
        Align a face to the reference template.

        Args:
            face_image: Input face image (BGR format)
            landmarks: 5-point facial landmarks
            validate_input: Whether to validate input parameters

        Returns:
            AlignedFace with aligned image and metadata

        Raises:
            ValueError: If input validation fails
        """
        if validate_input and not self._validate_landmarks(landmarks):
            raise ValueError("Invalid landmarks provided")

        if face_image.size == 0:
            raise ValueError("Empty face image provided")

        source_points = landmarks.to_array()

        width, height = self.output_size
        target_points = self.reference_landmarks.copy()
        target_points[:, 0] *= width
        target_points[:, 1] *= height

        transform_matrix = self._compute_similarity_transform(
            source_points, target_points
        )

        aligned = self._warp_image(face_image, transform_matrix, self.output_size)

        if self.enable_histogram_equalization:
            aligned = self._apply_histogram_equalization(aligned)

        logger.debug(
            f"Aligned face: {face_image.shape} -> {aligned.shape}, "
            f"matrix:\n{transform_matrix}"
        )

        return AlignedFace(
            image=aligned,
            landmarks=landmarks,
            transformation_matrix=transform_matrix,
            crop_size=self.output_size,
        )

    def align_from_array(
        self,
        face_image: np.ndarray,
        landmarks_array: np.ndarray,
    ) -> AlignedFace:
        """
        Align face from landmarks array.

        Args:
            face_image: Input face image (BGR)
            landmarks_array: Landmarks as numpy array (5, 2)

        Returns:
            AlignedFace result
        """
        landmarks = Landmark5Points.from_array(landmarks_array)
        return self.align(face_image, landmarks)

    def create_reference_template(
        self,
        size: Tuple[int, int],
    ) -> np.ndarray:
        """
        Create a visualization of the reference template.

        Args:
            size: Output image size (width, height)

        Returns:
            Template visualization as image
        """
        template = np.zeros((size[1], size[0], 3), dtype=np.uint8)

        width, height = size
        points = self.reference_landmarks.copy()
        points[:, 0] *= width
        points[:, 1] *= height

        colors = {
            "left_eye": (0, 255, 0),
            "right_eye": (0, 255, 0),
            "nose": (255, 0, 0),
            "left_mouth": (0, 0, 255),
            "right_mouth": (0, 0, 255),
        }

        point_names = ["left_eye", "right_eye", "nose", "left_mouth", "right_mouth"]
        for point, name in zip(points, point_names):
            x, y = int(point[0]), int(point[1])
            if 0 <= x < size[0] and 0 <= y < size[1]:
                cv2.circle(template, (x, y), 5, colors[name], -1)

        return template


def align_face(
    face_image: np.ndarray,
    landmarks: Landmark5Points,
    output_size: Tuple[int, int] = (224, 224),
    enable_histogram_equalization: bool = True,
) -> AlignedFace:
    """
    Convenience function for face alignment.

    Args:
        face_image: Input face image (BGR)
        landmarks: 5-point facial landmarks
        output_size: Output size (width, height)
        enable_histogram_equalization: Whether to apply histogram equalization

    Returns:
        AlignedFace result
    """
    aligner = FaceAligner(
        output_size=output_size,
        enable_histogram_equalization=enable_histogram_equalization,
    )
    return aligner.align(face_image, landmarks)


def align_face_from_array(
    face_image: np.ndarray,
    landmarks_array: np.ndarray,
    output_size: Tuple[int, int] = (224, 224),
) -> AlignedFace:
    """
    Convenience function for face alignment from array.

    Args:
        face_image: Input face image (BGR)
        landmarks_array: Landmarks as numpy array (5, 2)
        output_size: Output size (width, height)

    Returns:
        AlignedFace result
    """
    aligner = FaceAligner(output_size=output_size)
    return aligner.align_from_array(face_image, landmarks_array)
