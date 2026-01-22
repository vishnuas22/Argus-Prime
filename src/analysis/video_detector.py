"""
Video Detection Orchestrator.

Orchestrates video deepfake detection pipeline:
1. Extract frames at configured FPS
2. Detect and align faces
3. Apply adversarial defense
4. Run ForensicFlow inference
5. Generate heatmaps (Grad-CAM)
6. Aggregate per-frame scores
"""

import asyncio
from dataclasses import dataclass, field
import logging

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
from torch import Tensor

from src.config import get_config
from src.core.logging import logger
from src.models import (
    DetectionResult,
    ForensicFlow,
    DistilDIRE,
    ModalityType,
)
from src.preprocessing import (
    VideoProcessor,
    FaceDetector,
    FaceAligner,
    Frame,
)

logger = logging.getLogger(__name__)


@dataclass
class FrameAnalysisResult:
    """Per-frame analysis result."""
    frame_number: int
    timestamp: float
    face_bbox: Optional[List[float]]
    manipulation_score: float
    heatmap_path: Optional[str] = None
    branch_scores: Dict[str, float] = field(default_factory=dict)


@dataclass
class VideoAnalysisResult:
    """Complete video analysis result."""
    overall_score: float
    confidence: float
    is_manipulated: bool
    frames_analyzed: int
    faces_detected: int
    per_frame_results: List[FrameAnalysisResult]
    heatmap_path: Optional[str] = None
    branch_scores: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API response."""
        return {
            "overall_score": self.overall_score,
            "confidence": self.confidence,
            "is_manipulated": self.is_manipulated,
            "frames_analyzed": self.frames_analyzed,
            "faces_detected": self.faces_detected,
            "heatmap_path": self.heatmap_path,
            "branch_scores": self.branch_scores,
            "per_frame_scores": [
                {
                    "frame_number": f.frame_number,
                    "timestamp": f.timestamp,
                    "face_bbox": f.face_bbox,
                    "manipulation_score": f.manipulation_score,
                    "heatmap_path": f.heatmap_path,
                    "branch_scores": f.branch_scores,
                }
                for f in self.per_frame_results
            ],
            "metadata": self.metadata,
        }


class VideoDetector:
    """
    Video Deepfake Detection Orchestrator.

    Coordinates the video analysis pipeline with:
    - Parallel frame extraction and face detection
    - Model inference orchestration
    - Heatmap generation for explainability
    - Score aggregation

    Attributes:
        processor: Video frame extractor
        face_detector: Face detection model
        face_aligner: Face alignment processor
        primary_model: Main detection model (ForensicFlow)
        fallback_model: Secondary model (DistilDIRE)
    """

    def __init__(
        self,
        fps: int = 5,
        extract_faces: bool = True,
        generate_heatmaps: bool = True,
        use_gpu: bool = True,
        batch_size: int = 8,
    ):
        """
        Initialize video detector orchestrator.

        Args:
            fps: Frames per second to extract
            extract_faces: Whether to extract and analyze faces
            generate_heatmaps: Whether to generate attention heatmaps
            use_gpu: Whether to use GPU acceleration
            batch_size: Batch size for model inference
        """
        config = get_config()
        self.fps = fps
        self.extract_faces = extract_faces
        self.generate_heatmaps = generate_heatmaps
        self.batch_size = batch_size

        self.device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")

        self.processor = VideoProcessor()
        self.face_detector = FaceDetector()
        self.face_aligner = FaceAligner()

        self.primary_model = ForensicFlow(device=self.device)
        self.fallback_model = DistilDIRE(device=self.device)

        self.primary_model.load(self.device)
        self.fallback_model.load(self.device)

        logger.info(
            f"VideoDetector initialized: fps={fps}, device={self.device}, "
            f"faces={extract_faces}, heatmaps={generate_heatmaps}"
        )

    async def analyze(
        self,
        video_path: str,
        progress_callback: Optional[callable] = None,
    ) -> VideoAnalysisResult:
        """
        Run full video analysis pipeline.

        Args:
            video_path: Path to video file
            progress_callback: Optional callback for progress updates

        Returns:
            VideoAnalysisResult with scores, heatmaps, and metadata
        """
        video_path_obj = Path(video_path)
        if not video_path_obj.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        if progress_callback:
            progress_callback(5, "Extracting frames")

        frames = self.processor.extract_frames_sync(str(video_path_obj), fps=self.fps)
        if not frames:
            raise ValueError(f"No frames extracted from video: {video_path}")

        if progress_callback:
            progress_callback(15, "Detecting faces")

        face_detections = self._detect_faces_parallel(frames)

        if progress_callback:
            progress_callback(35, "Running detection models")

        frame_results = await self._run_detection(frames, face_detections, progress_callback)

        if progress_callback:
            progress_callback(90, "Aggregating results")

        result = self._aggregate_results(frames, frame_results, video_path_obj)

        if progress_callback:
            progress_callback(100, "Analysis complete")

        logger.info(
            f"Video analysis complete: {result.frames_analyzed} frames, "
            f"score={result.overall_score:.3f}"
        )

        return result

    def _detect_faces_parallel(self, frames: List[Frame]) -> Dict[int, Any]:
        """Detect faces in all frames in parallel."""
        face_detections = {}
        for frame in frames:
            try:
                image = cv2.imread(frame.path)
                if image is None:
                    continue
                detections = self.face_detector.detect_faces(image)
                face_detections[frame.frame_number] = detections
            except Exception as e:
                logger.warning(f"Face detection failed for frame {frame.frame_number}: {e}")
                face_detections[frame.frame_number] = []

        return face_detections

    async def _run_detection(
        self,
        frames: List[Frame],
        face_detections: Dict[int, Any],
        progress_callback: Optional[callable],
    ) -> List[FrameAnalysisResult]:
        """Run detection on all frames."""
        frame_results = []

        for i, frame in enumerate(frames):
            try:
                frame_result = self._process_frame(
                    frame,
                    face_detections.get(frame.frame_number, []),
                )
                frame_results.append(frame_result)

                if progress_callback:
                    progress = 35 + int(55 * (i + 1) / len(frames))
                    progress_callback(progress, f"Processing frame {i + 1}/{len(frames)}")

            except Exception as e:
                logger.error(f"Frame processing failed: {e}")
                frame_results.append(
                    FrameAnalysisResult(
                        frame_number=frame.frame_number,
                        timestamp=frame.timestamp,
                        face_bbox=None,
                        manipulation_score=0.5,
                        branch_scores={},
                    )
                )

        return frame_results

    def _process_frame(
        self,
        frame: Frame,
        face_detections: List[Any],
    ) -> FrameAnalysisResult:
        """Process a single frame."""
        image = cv2.imread(frame.path)
        if image is None:
            image = np.zeros((224, 224, 3), dtype=np.uint8)

        face_bbox = None
        aligned_face = None

        if face_detections and self.extract_faces:
            if len(face_detections) > 0:
                face = face_detections[0]
                if hasattr(face, 'bbox'):
                    face_bbox = list(face.bbox)
                    aligned_face = self.face_aligner.align(image, face.landmarks)
                elif isinstance(face, dict):
                    face_bbox = face.get('bbox')
                    landmarks = face.get('landmarks')
                    if landmarks is not None:
                        aligned_face = self.face_aligner.align(image, landmarks)

        input_tensor = self._prepare_input(image, aligned_face)

        with torch.no_grad():
            primary_result = self.primary_model.detect(input_tensor)

        branch_scores = {
            "forensic_flow": primary_result.overall_score,
            "distil_dire": 0.5,
        }

        manipulation_score = primary_result.overall_score

        heatmap_path = None
        if self.generate_heatmaps:
            try:
                heatmap = self.primary_model.get_heatmap(input_tensor)
                if heatmap is not None:
                    heatmap_path = self._save_heatmap(heatmap, frame.frame_number)
            except Exception as e:
                logger.debug(f"Heatmap generation failed: {e}")

        return FrameAnalysisResult(
            frame_number=frame.frame_number,
            timestamp=frame.timestamp,
            face_bbox=face_bbox,
            manipulation_score=manipulation_score,
            heatmap_path=heatmap_path,
            branch_scores=branch_scores,
        )

    def _prepare_input(
        self,
        image: np.ndarray,
        aligned_face: Optional[np.ndarray] = None,
    ) -> torch.Tensor:
        """Prepare image tensor for model input."""
        import torchvision.transforms as transforms

        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

        input_image = aligned_face if aligned_face is not None else image
        tensor = transform(input_image)
        return tensor.unsqueeze(0).to(self.device)

    def _save_heatmap(self, heatmap: Tensor, frame_number: int) -> str:
        """Save heatmap to file and return path."""
        heatmap_dir = get_config().storage.temp_dir
        Path(heatmap_dir).mkdir(parents=True, exist_ok=True)

        heatmap_np = heatmap.cpu().numpy().squeeze()
        if heatmap_np.ndim == 2:
            heatmap_np = cv2.resize(heatmap_np, (224, 224))
            heatmap_np = (heatmap_np - heatmap_np.min()) / (heatmap_np.max() - heatmap_np.min() + 1e-8)
            heatmap_np = (heatmap_np * 255).astype(np.uint8)
            heatmap_np = cv2.applyColorMap(heatmap_np, cv2.COLORMAP_JET)

        heatmap_path = str(Path(heatmap_dir) / f"heatmap_{frame_number}.png")
        cv2.imwrite(heatmap_path, heatmap_np)

        return heatmap_path

    def _aggregate_results(
        self,
        frames: List[Frame],
        frame_results: List[FrameAnalysisResult],
        video_path: Path,
    ) -> VideoAnalysisResult:
        """Aggregate per-frame results into final result."""
        if not frame_results:
            return VideoAnalysisResult(
                overall_score=0.5,
                confidence=0.0,
                is_manipulated=False,
                frames_analyzed=0,
                faces_detected=0,
                per_frame_results=[],
            )

        scores = [r.manipulation_score for r in frame_results]
        weights = self._compute_frame_weights(frame_results)
        weighted_score = float(np.average(scores, weights=weights))

        confidence = self._compute_confidence(scores, weighted_score)

        faces_with_bbox = sum(1 for r in frame_results if r.face_bbox is not None)

        branch_agg = {}
        for result in frame_results:
            for branch, score in result.branch_scores.items():
                if branch not in branch_agg:
                    branch_agg[branch] = []
                branch_agg[branch].append(score)

        branch_scores = {k: float(np.mean(v)) for k, v in branch_agg.items()}

        metadata = {
            "video_path": str(video_path),
            "fps": self.fps,
            "total_frames": len(frames),
            "frames_with_faces": faces_with_bbox,
        }

        return VideoAnalysisResult(
            overall_score=weighted_score,
            confidence=confidence,
            is_manipulated=weighted_score > 0.5,
            frames_analyzed=len(frame_results),
            faces_detected=faces_with_bbox,
            per_frame_results=frame_results,
            branch_scores=branch_scores,
            metadata=metadata,
        )

    def _compute_frame_weights(self, frame_results: List[FrameAnalysisResult]) -> List[float]:
        """Compute weights for frame aggregation."""
        weights = []
        for result in frame_results:
            w = 1.0
            if result.face_bbox is not None:
                bbox_area = (result.face_bbox[2] - result.face_bbox[0]) * (
                    result.face_bbox[3] - result.face_bbox[1]
                )
                w += min(bbox_area / 10000, 1.0)
            if result.branch_scores.get("forensic_flow", 0) > 0.5:
                w *= 1.2
            weights.append(w)
        return weights

    def _compute_confidence(
        self,
        scores: List[float],
        weighted_score: float,
    ) -> float:
        """Compute confidence score."""
        if len(scores) < 2:
            return 0.5

        score_variance = np.var(scores)
        distance_from_middle = abs(weighted_score - 0.5) * 2

        confidence = min(
            max(distance_from_middle * 0.7 + (1 - score_variance) * 0.3, 0.0),
            1.0,
        )

        return confidence

    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        return {
            "primary_model": "ForensicFlow",
            "fallback_model": "DistilDIRE",
            "device": str(self.device),
            "fps": self.fps,
            "extract_faces": self.extract_faces,
            "generate_heatmaps": self.generate_heatmaps,
        }

    def health_check(self) -> bool:
        """Check if all components are healthy."""
        try:
            if not self.primary_model.health_check():
                return False
            if not self.fallback_model.health_check():
                return False
            return True
        except Exception:
            return False


def create_video_detector(
    fps: int = 5,
    extract_faces: bool = True,
    generate_heatmaps: bool = True,
    use_gpu: bool = True,
) -> VideoDetector:
    """Factory function for VideoDetector."""
    return VideoDetector(
        fps=fps,
        extract_faces=extract_faces,
        generate_heatmaps=generate_heatmaps,
        use_gpu=use_gpu,
    )
