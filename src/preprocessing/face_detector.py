"""
Face detection module for Argus Prime.

Provides multi-face detection using MTCNN (Multi-task Cascaded CNN)
with support for GPU acceleration and batch processing.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Any
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torchvision.transforms import functional as TF

from src.config import get_config
from src.core.logging import logger


@dataclass
class FaceBox:
    """
    Internal representation of a detected face.

    Attributes:
        bbox: Bounding box coordinates (x1, y1, x2, y2)
        landmarks: 5-point facial landmarks (eyes, nose, mouth corners)
        confidence: Detection confidence score (0-1)
        aligned_image: Aligned face tensor (optional)
    """
    bbox: Tuple[float, float, float, float]
    landmarks: Tuple[Tuple[float, float], ...]
    confidence: float
    aligned_image: Optional[np.ndarray] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert FaceBox to dictionary for serialization."""
        return {
            "bbox": list(self.bbox),
            "landmarks": [list(lm) for lm in self.landmarks],
            "confidence": self.confidence,
        }


class MTCNN:
    """
    MTCNN (Multi-task Cascaded CNN) face detector.

    This class implements the MTCNN architecture for efficient and accurate
    face detection with 5-point landmark extraction.

    The architecture consists of three stages:
    - P-Net: Proposal network for fast initial detection
    - R-Net: Refine network for accurate filtering
    - O-Net: Output network for final detection and landmarks

    Attributes:
        device: PyTorch device (CUDA/MPS/CPU)
        p_thresh: Threshold for P-Net detection
        r_thresh: Threshold for R-Net detection
        o_thresh: Threshold for O-Net detection
        min_face_size: Minimum face size to detect

    Example:
        >>> detector = MTCNN()
        >>> faces = detector.detect(frame_array)
        >>> aligned = detector.align_face(frame_array, faces[0])
    """

    def __init__(
        self,
        device: Optional[torch.device] = None,
        p_thresh: float = 0.6,
        r_thresh: float = 0.7,
        o_thresh: float = 0.8,
        min_face_size: int = 20,
        model_path: Optional[str] = None,
    ):
        """
        Initialize MTCNN face detector.

        Args:
            device: PyTorch device (auto-detected if None)
            p_thresh: Detection threshold for P-Net (default: 0.6)
            r_thresh: Detection threshold for R-Net (default: 0.7)
            o_thresh: Detection threshold for O-Net (default: 0.8)
            min_face_size: Minimum face size in pixels (default: 20)
            model_path: Path to model weights (auto-downloaded if None)
        """
        self.device = device or self._get_device()
        self.p_thresh = p_thresh
        self.r_thresh = r_thresh
        self.o_thresh = o_thresh
        self.min_face_size = min_face_size

        # Load models
        self._load_models(model_path)

        logger.info(
            f"MTCNN initialized: device={self.device}, "
            f"p_thresh={self.p_thresh}, r_thresh={self.r_thresh}, "
            f"o_thresh={self.o_thresh}, min_face={self.min_face_size}"
        )

    def _get_device(self) -> torch.device:
        """Auto-detect best available device."""
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")

    def _load_models(self, model_path: Optional[str] = None) -> None:
        """
        Load MTCNN model weights.

        Args:
            model_path: Path to model weights file
        """
        # Try to load pretrained weights
        try:
            from torchvision.models.detection import mtcnn as tv_mtcnn

            # Use torchvision's MTCNN implementation
            self.mtcnn = tv_mtcnn.MTCNN(
                image_size=160,
                margin=0,
                device=self.device,
                transform_func=TF.to_tensor,
            ).to(self.device)

            self.mtcnn.eval()
            logger.info("Loaded torchvision MTCNN model")

        except ImportError:
            # Fallback to custom implementation if torchvision version not available
            logger.warning("torchvision MTCNN not available, using custom implementation")
            self._build_custom_models()

    def _build_custom_models(self) -> None:
        """Build custom MTCNN models (P-Net, R-Net, O-Net)."""
        # P-Net: 12x12 proposal network
        self.p_net = self._create_p_net().to(self.device)
        self.p_net.eval()

        # R-Net: 24x24 refine network
        self.r_net = self._create_r_net().to(self.device)
        self.r_net.eval()

        # O-Net: 48x48 output network
        self.o_net = self._create_o_net().to(self.device)
        self.o_net.eval()

    def _create_p_net(self) -> torch.nn.Module:
        """Create P-Net model."""
        return torch.nn.Sequential(
            torch.nn.Conv2d(3, 10, kernel_size=3, stride=1, padding=1),
            torch.nn.PReLU(10),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Conv2d(10, 16, kernel_size=3, stride=1, padding=1),
            torch.nn.PReLU(16),
            torch.nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            torch.nn.PReLU(32),
        )

    def _create_r_net(self) -> torch.nn.Module:
        """Create R-Net model."""
        return torch.nn.Sequential(
            torch.nn.Conv2d(3, 28, kernel_size=5, stride=1, padding=2),
            torch.nn.PReLU(28),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Conv2d(28, 48, kernel_size=3, stride=1, padding=1),
            torch.nn.PReLU(48),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Conv2d(48, 64, kernel_size=3, stride=1, padding=1),
            torch.nn.PReLU(64),
            torch.nn.Flatten(),
            torch.nn.Linear(64 * 3 * 3, 128),
            torch.nn.PReLU(128),
        )

    def _create_o_net(self) -> torch.nn.Module:
        """Create O-Net model."""
        return torch.nn.Sequential(
            torch.nn.Conv2d(3, 32, kernel_size=5, stride=1, padding=2),
            torch.nn.PReLU(32),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            torch.nn.PReLU(64),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Conv2d(64, 80, kernel_size=3, stride=1, padding=1),
            torch.nn.PReLU(80),
            torch.nn.Flatten(),
            torch.nn.Linear(80 * 4 * 4, 256),
            torch.nn.PReLU(256),
        )

    def detect(
        self,
        image: np.ndarray,
        thresholds: Optional[Dict[str, float]] = None,
    ) -> List[FaceBox]:
        """
        Detect faces in a single image.

        Args:
            image: Input image as numpy array (H, W, 3) in RGB format
            thresholds: Optional custom thresholds for each stage

        Returns:
            List of FaceBox objects with detected faces
        """
        if image is None or image.size == 0:
            logger.warning("Empty image provided for face detection")
            return []

        # Convert to RGB if needed
        if image.shape[2] == 4:
            image = image[:, :, :3]
        elif image.shape[2] != 3:
            raise ValueError(f"Invalid image shape: {image.shape}. Expected (H, W, 3)")

        h, w = image.shape[:2]

        # Use torchvision MTCNN if available
        if hasattr(self, 'mtcnn'):
            return self._detect_with_torchvision(image, thresholds)

        # Use custom implementation
        return self._detect_with_custom(image, thresholds)

    def _detect_with_torchvision(
        self,
        image: np.ndarray,
        thresholds: Optional[Dict[str, float]] = None,
    ) -> List[FaceBox]:
        """Detect faces using torchvision MTCNN."""
        # Convert to tensor
        img_tensor = TF.to_tensor(image).unsqueeze(0).to(self.device)

        # Detect faces
        with torch.no_grad():
            boxes, probs = self.mtcnn(img_tensor, landmarks=True)

        faces = []
        if boxes is not None:
            for i, (box, landm, prob) in enumerate(zip(boxes, probs[0] if len(probs) > 0 else [], boxes)):
                # Convert to numpy
                box = box.cpu().numpy()
                landm = landm.cpu().numpy() if hasattr(landm, 'cpu') else np.array(landm)

                # Validate box coordinates
                x1, y1, x2, y2 = box
                if x2 > x1 and y2 > y1:
                    # Normalize confidence
                    confidence = float(prob) if isinstance(prob, torch.Tensor) else float(prob)

                    # Extract 5 landmarks
                    landmarks = (
                        (float(landm[0]), float(landm[5])),   # left eye
                        (float(landm[1]), float(landm[6])),   # right eye
                        (float(landm[2]), float(landm[7])),   # nose
                        (float(landm[3]), float(landm[8])),   # left mouth
                        (float(landm[4]), float(landm[9])),   # right mouth
                    )

                    faces.append(FaceBox(
                        bbox=(float(x1), float(y1), float(x2), float(y2)),
                        landmarks=landmarks,
                        confidence=confidence,
                    ))

        return faces

    def _detect_with_custom(
        self,
        image: np.ndarray,
        thresholds: Optional[Dict[str, float]] = None,
    ) -> List[FaceBox]:
        """Detect faces using custom MTCNN implementation."""
        # Placeholder for custom implementation
        # In production, this would implement the full MTCNN pipeline
        logger.warning("Custom MTCNN implementation not fully available")
        return []

    def detect_batch(
        self,
        images: List[np.ndarray],
        thresholds: Optional[Dict[str, float]] = None,
    ) -> List[List[FaceBox]]:
        """
        Detect faces in a batch of images.

        Args:
            images: List of images as numpy arrays (H, W, 3)
            thresholds: Optional custom thresholds for each stage

        Returns:
            List of lists of FaceBox objects for each image
        """
        results = []
        for image in images:
            faces = self.detect(image, thresholds)
            results.append(faces)

        logger.info(f"Batch detection: {len(images)} images, {sum(len(f) for f in results)} faces")
        return results

    def align_face(
        self,
        image: np.ndarray,
        face: FaceBox,
        target_size: Tuple[int, int] = (224, 224),
    ) -> np.ndarray:
        """
        Align a detected face using 5-point landmarks.

        Args:
            image: Original image containing the face
            face: FaceBox with detection results
            target_size: Target size for aligned face (default: 224x224)

        Returns:
            Aligned face image as numpy array
        """
        import cv2

        # Get landmark points
        landmarks = np.array(face.landmarks)

        # Reference template (based on FFHQ/CelebA)
        ref_template = np.array([
            [30.2946, 51.6963],
            [65.5318, 51.5014],
            [48.0252, 71.7366],
            [33.5493, 92.3655],
            [62.7299, 92.2041],
        ])

        # Scale reference to target size
        scale = min(target_size) / 112.0
        ref_template = ref_template * scale

        # Compute similarity transform
        h, w = target_size
        dst_points = np.array([
            [w * 0.5, h * 0.3],
            [w * 0.5, h * 0.7],
            [w * 0.5, h * 0.5],
            [w * 0.3, h * 0.85],
            [w * 0.7, h * 0.85],
        ])

        # Calculate transformation matrix
        matrix, _ = cv2.estimateAffinePartial2D(landmarks[:3], dst_points[:3])

        if matrix is None:
            logger.warning("Failed to compute alignment transform")
            return np.zeros((*target_size, 3), dtype=np.uint8)

        # Warp image
        aligned = cv2.warpAffine(
            image,
            matrix,
            (target_size[1], target_size[0]),
            borderValue=0,
        )

        return aligned

    def validate_image(self, image: np.ndarray) -> tuple[bool, str]:
        """
        Validate that the input is a valid image for face detection.

        Args:
            image: Input image as numpy array

        Returns:
            Tuple of (is_valid, error_message)
        """
        if image is None:
            return False, "Image is None"

        if not isinstance(image, np.ndarray):
            return False, f"Expected numpy array, got {type(image)}"

        if image.ndim != 3:
            return False, f"Expected 3D array (H, W, 3), got {image.ndim}D"

        h, w = image.shape[:2]
        if h < 64 or w < 64:
            return False, f"Image too small: {w}x{h}. Minimum size: 64x64"

        if image.shape[2] not in [3, 4]:
            return False, f"Expected 3 or 4 channels, got {image.shape[2]}"

        return True, ""


class FaceDetector:
    """
    High-level face detection interface.

    This class provides a simplified interface for face detection
    with automatic preprocessing and result formatting.

    Attributes:
        mtcnn: MTCNN detector instance
        batch_size: Batch size for batch processing
        device: PyTorch device for inference

    Example:
        >>> detector = FaceDetector()
        >>> faces = detector.detect_faces(frame_array)
        >>> aligned = detector.align(faces[0])
    """

    def __init__(
        self,
        device: Optional[torch.device] = None,
        batch_size: int = 8,
        min_face_size: int = 20,
    ):
        """
        Initialize face detector.

        Args:
            device: PyTorch device (auto-detected if None)
            batch_size: Batch size for processing
            min_face_size: Minimum face size to detect
        """
        self.device = device or self._get_device()
        self.batch_size = batch_size

        self.mtcnn = MTCNN(
            device=self.device,
            min_face_size=min_face_size,
        )

        logger.info(
            f"FaceDetector initialized: device={self.device}, batch_size={batch_size}"
        )

    def _get_device(self) -> torch.device:
        """Auto-detect best available device."""
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")

    def detect_faces(
        self,
        image: np.ndarray,
        return_aligned: bool = False,
        target_size: Tuple[int, int] = (224, 224),
    ) -> List[FaceBox]:
        """
        Detect faces in an image.

        Args:
            image: Input image as numpy array (H, W, 3) in RGB format
            return_aligned: Whether to return aligned face images
            target_size: Size for aligned faces

        Returns:
            List of FaceBox objects
        """
        # Validate input
        is_valid, error_msg = self.mtcnn.validate_image(image)
        if not is_valid:
            logger.warning(f"Invalid image: {error_msg}")
            return []

        # Detect faces
        faces = self.mtcnn.detect(image)

        # Optionally align faces
        if return_aligned and faces:
            for face in faces:
                aligned = self.mtcnn.align_face(image, face, target_size)
                face.aligned_image = aligned

        return faces

    def detect_faces_batch(
        self,
        images: List[np.ndarray],
        return_aligned: bool = False,
        target_size: Tuple[int, int] = (224, 224),
    ) -> List[List[FaceBox]]:
        """
        Detect faces in a batch of images.

        Args:
            images: List of input images
            return_aligned: Whether to return aligned face images
            target_size: Size for aligned faces

        Returns:
            List of lists of FaceBox objects
        """
        if not images:
            return []

        # Process in batches
        results = self.mtcnn.detect_batch(images)

        # Optionally align faces
        if return_aligned:
            for i, (image, faces) in enumerate(zip(images, results)):
                for face in faces:
                    aligned = self.mtcnn.align_face(image, face, target_size)
                    face.aligned_image = aligned

        return results

    def get_largest_face(self, image: np.ndarray) -> Optional[FaceBox]:
        """
        Detect the largest face in an image.

        Args:
            image: Input image

        Returns:
            FaceBox of largest face or None if no faces detected
        """
        faces = self.detect_faces(image)

        if not faces:
            return None

        # Find largest face by area
        largest = max(
            faces,
            key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])
        )

        return largest


def detect_faces(
    image: np.ndarray,
    device: Optional[torch.device] = None,
) -> List[FaceBox]:
    """
    Convenience function for face detection.

    Args:
        image: Input image as numpy array (H, W, 3)
        device: PyTorch device (auto-detected if None)

    Returns:
        List of FaceBox objects
    """
    detector = FaceDetector(device=device)
    return detector.detect_faces(image)


def align_face(
    image: np.ndarray,
    face: FaceBox,
    target_size: Tuple[int, int] = (224, 224),
) -> np.ndarray:
    """
    Convenience function for face alignment.

    Args:
        image: Original image containing the face
        face: FaceBox with detection results
        target_size: Target size for aligned face

    Returns:
        Aligned face image
    """
    detector = FaceDetector()
    return detector.mtcnn.align_face(image, face, target_size)
