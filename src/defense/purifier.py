"""
DG-GAN Purifier - Generator-based Adversarial Purification.

Implements DG-GAN (Defense Generator - Generative Adversarial Network) for
adaptive purification of adversarial inputs.

Architecture:
- Encoder: Maps input to latent space
- Generator: Reconstructs from latent
- Discriminator: Distinguishes clean vs adversarial
- Training: Purify → Detect loop

Outperforms adversarial training on iterative attacks (PGD, DeepFool, C&W).
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from src.config import get_config
from src.core.logging import logger


@dataclass
class PurifyResult:
    """Result from DG-GAN purification."""
    purified: np.ndarray
    is_adversarial: bool
    purification_applied: bool
    reconstruction_error: float
    discriminator_score: float
    confidence: float
    details: Dict[str, Any] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "purified": self.purified.tolist() if isinstance(self.purified, np.ndarray) else self.purified,
            "is_adversarial": self.is_adversarial,
            "purification_applied": self.purification_applied,
            "reconstruction_error": self.reconstruction_error,
            "discriminator_score": self.discriminator_score,
            "confidence": self.confidence,
            "details": self.details or {},
        }


class Encoder(nn.Module):
    """Encoder network that maps input to latent space."""

    def __init__(self, in_channels: int = 3, latent_dim: int = 256):
        super().__init__()
        self.in_channels = in_channels
        self.latent_dim = latent_dim

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 4, 2, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, latent_dim, 4, 1, 0),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        return x


class Generator(nn.Module):
    """Generator network that reconstructs from latent space."""

    def __init__(self, latent_dim: int = 256, out_channels: int = 3):
        super().__init__()
        self.latent_dim = latent_dim
        self.out_channels = out_channels

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 512, 4, 1, 0),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(512, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, out_channels, 4, 2, 1),
            nn.Tanh(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.decoder(z)
        return x


class Discriminator(nn.Module):
    """Discriminator network for adversarial training."""

    def __init__(self, in_channels: int = 3):
        super().__init__()
        self.in_channels = in_channels

        self.discriminator = nn.Sequential(
            nn.Conv2d(in_channels, 64, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 4, 2, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, 4, 1, 0),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.discriminator(x)
        return x


class DGGANPurifier:
    """
    DG-GAN Based Adversarial Purifier.

    Uses a generator-discriminator architecture to purify adversarial inputs:
    - Encoder maps input to latent space
    - Generator reconstructs purified version
    - Discriminator distinguishes clean from adversarial
    - Training loop: Purify → Detect → Improve

    Attributes:
        latent_dim: Dimension of latent space
        device: Computation device
        purification_steps: Number of purification iterations
        threshold: Threshold for adversarial detection
    """

    def __init__(
        self,
        latent_dim: int = 256,
        device: Optional[torch.device] = None,
        purification_steps: int = 1,
        threshold: float = 0.5,
        load_pretrained: bool = False,
    ):
        """
        Initialize DG-GAN purifier.

        Args:
            latent_dim: Dimension of latent space
            device: Computation device
            purification_steps: Number of purification iterations
            threshold: Threshold for adversarial detection
            load_pretrained: Whether to load pretrained weights
        """
        self.latent_dim = latent_dim
        self.device = device or torch.device("cpu")
        self.purification_steps = purification_steps
        self.threshold = threshold

        self.encoder = Encoder(in_channels=3, latent_dim=latent_dim).to(self.device)
        self.generator = Generator(latent_dim=latent_dim, out_channels=3).to(self.device)
        self.discriminator = Discriminator(in_channels=3).to(self.device)

        if load_pretrained:
            self._load_pretrained_weights()

        self.encoder.eval()
        self.generator.eval()
        self.discriminator.eval()

        config = get_config()
        self.defense_config = config.defense

        logger.info(
            f"DGGANPurifier initialized: latent_dim={latent_dim}, "
            f"purification_steps={purification_steps}, device={self.device}"
        )

    def _load_pretrained_weights(self):
        """Load pretrained DG-GAN weights if available."""
        try:
            model_path = get_config().models.get("dggan_path", "models/dggan_purifier.pt")
            if torch.cuda.is_available():
                state_dict = torch.load(model_path, map_location=self.device)
            else:
                state_dict = torch.load(model_path, map_location=self.device)
            self.encoder.load_state_dict(state_dict["encoder"])
            self.generator.load_state_dict(state_dict["generator"])
            self.discriminator.load_state_dict(state_dict["discriminator"])
            logger.info("Loaded pretrained DG-GAN weights")
        except Exception as e:
            logger.warning(f"Could not load pretrained DG-GAN weights: {e}")

    @torch.no_grad()
    def purify(self, image: np.ndarray) -> Tuple[np.ndarray, PurifyResult]:
        """
        Purify a single image using DG-GAN.

        Args:
            image: Input image (H, W, 3) numpy array, values 0-255

        Returns:
            Tuple of (purified_image, PurifyResult)
        """
        original_tensor = self._image_to_tensor(image).unsqueeze(0)
        latent = self.encoder(original_tensor)
        purified_tensor = self.generator(latent)

        purified_np = self._tensor_to_image(purified_tensor.squeeze(0))

        reconstruction_error = float(F.mse_loss(purified_tensor, original_tensor).item())

        discriminator_input = torch.cat([original_tensor, purified_tensor], dim=0)
        discriminator_output = self.discriminator(discriminator_input)
        disc_scores = torch.sigmoid(discriminator_output).view(-1)
        discriminator_score = float(disc_scores[1].item())

        is_adversarial = discriminator_score > self.threshold

        confidence = abs(discriminator_score - 0.5) * 2

        result = PurifyResult(
            purified=purified_np,
            is_adversarial=is_adversarial,
            purification_applied=True,
            reconstruction_error=reconstruction_error,
            discriminator_score=discriminator_score,
            confidence=confidence,
            details={
                "latent_shape": list(latent.shape),
                "original_shape": list(original_tensor.shape),
                "purification_steps": self.purification_steps,
                "threshold": self.threshold,
            },
        )

        return purified_np, result

    @torch.no_grad()
    def purify_batch(self, images: List[np.ndarray]) -> Tuple[List[np.ndarray], List[PurifyResult]]:
        """
        Purify a batch of images.

        Args:
            images: List of images (H, W, 3) numpy arrays

        Returns:
            Tuple of (purified_images, list of PurifyResult)
        """
        if not images:
            return [], []

        original_tensors = torch.stack([self._image_to_tensor(img) for img in images])
        original_tensors = original_tensors.to(self.device)

        latents = self.encoder(original_tensors)
        purified_tensors = self.generator(latents)

        purified_images = [self._tensor_to_image(t) for t in purified_tensors]

        reconstruction_errors = F.mse_loss(purified_tensors, original_tensors, reduction="none")
        reconstruction_errors = reconstruction_errors.view(reconstruction_errors.size(0), -1).mean(dim=1)

        discriminator_input = torch.cat([original_tensors, purified_tensors], dim=0)
        discriminator_output = self.discriminator(discriminator_input)
        disc_scores = torch.sigmoid(discriminator_output).view(-1)
        disc_scores = disc_scores.view(2, -1).mean(dim=0)

        results = []
        for i, (purified_img, recon_err, disc_score) in enumerate(
            zip(purified_images, reconstruction_errors, disc_scores)
        ):
            is_adversarial = disc_score.item() > self.threshold
            confidence = abs(disc_score.item() - 0.5) * 2

            results.append(
                PurifyResult(
                    purified=purified_img,
                    is_adversarial=is_adversarial,
                    purification_applied=True,
                    reconstruction_error=recon_err.item(),
                    discriminator_score=disc_score.item(),
                    confidence=confidence,
                    details={"image_index": i},
                )
            )

        return purified_images, results

    @torch.no_grad()
    def detect_adversarial(self, image: np.ndarray) -> bool:
        """
        Detect if an image is adversarial without purifying.

        Args:
            image: Input image (H, W, 3) numpy array

        Returns:
            True if image is detected as adversarial
        """
        original_tensor = self._image_to_tensor(image).unsqueeze(0)
        latent = self.encoder(original_tensor)
        purified_tensor = self.generator(latent)

        discriminator_input = torch.cat([original_tensor, purified_tensor], dim=0)
        discriminator_output = self.discriminator(discriminator_input)
        disc_scores = torch.sigmoid(discriminator_output).view(-1)
        disc_score = disc_scores[0].item()

        return disc_score > self.threshold

    def _image_to_tensor(self, image: np.ndarray) -> torch.Tensor:
        """Convert numpy image to tensor."""
        image = image.astype(np.float32) / 255.0
        image = (image - 0.5) / 0.5
        image = torch.from_numpy(image).permute(2, 0, 1)
        return image.to(self.device)

    def _tensor_to_image(self, tensor: torch.Tensor) -> np.ndarray:
        """Convert tensor to numpy image."""
        tensor = tensor.squeeze(0).detach().cpu()
        tensor = tensor * 0.5 + 0.5
        tensor = torch.clamp(tensor, 0, 1)
        image = (tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        return image

    def get_purifier_info(self) -> Dict[str, Any]:
        """Get purifier configuration and status."""
        return {
            "latent_dim": self.latent_dim,
            "purification_steps": self.purification_steps,
            "threshold": self.threshold,
            "device": str(self.device),
            "encoder_params": sum(p.numel() for p in self.encoder.parameters()),
            "generator_params": sum(p.numel() for p in self.generator.parameters()),
            "discriminator_params": sum(p.numel() for p in self.discriminator.parameters()),
        }


def create_dggan_purifier() -> DGGANPurifier:
    """Factory function for DGGANPurifier."""
    config = get_config()
    return DGGANPurifier(
        latent_dim=config.defense.get("dggan_latent_dim", 256),
        purification_steps=config.defense.get("dggan_purification_steps", 1),
        threshold=config.defense.get("dggan_threshold", 0.5),
        load_pretrained=config.defense.get("dggan_load_pretrained", False),
    )
