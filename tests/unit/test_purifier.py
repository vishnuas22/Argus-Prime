"""
Unit tests for DG-GAN Purifier - Generator-based Adversarial Purification.
"""

import pytest
import torch
import numpy as np

from src.defense import DGGANPurifier, PurifyResult


class TestDGGANPurifier:
    """Test DG-GAN Purifier."""

    def test_initialization(self):
        """Test purifier initialization."""
        purifier = DGGANPurifier(
            latent_dim=128,
            purification_steps=2,
            threshold=0.6,
        )

        assert purifier.latent_dim == 128
        assert purifier.purification_steps == 2
        assert purifier.threshold == 0.6
        assert purifier.device.type in ["cpu", "cuda"]

    def test_purify_single_image(self):
        """Test purification of a single image."""
        purifier = DGGANPurifier(latent_dim=64)

        image = (np.random.rand(64, 64, 3) * 255).astype(np.uint8)

        purified, result = purifier.purify(image)

        assert isinstance(result, PurifyResult)
        assert purified.shape == image.shape
        assert 0 <= result.reconstruction_error <= 1.0
        assert 0 <= result.discriminator_score <= 1.0
        assert 0 <= result.confidence <= 1.0
        assert isinstance(result.is_adversarial, bool)
        assert result.purification_applied is True

    def test_purify_batch_images(self):
        """Test batch purification of images."""
        purifier = DGGANPurifier(latent_dim=64)

        images = [(np.random.rand(64, 64, 3) * 255).astype(np.uint8) for _ in range(4)]

        purified_images, results = purifier.purify_batch(images)

        assert len(purified_images) == 4
        assert len(results) == 4
        assert all(isinstance(r, PurifyResult) for r in results)
        assert all(p.shape == (64, 64, 3) for p in purified_images)

    def test_detect_adversarial(self):
        """Test adversarial detection without purification."""
        purifier = DGGANPurifier(latent_dim=64)

        image = (np.random.rand(64, 64, 3) * 255).astype(np.uint8)

        is_adversarial = purifier.detect_adversarial(image)

        assert isinstance(is_adversarial, bool)

    def test_purify_result_to_dict(self):
        """Test PurifyResult serialization."""
        result = PurifyResult(
            purified=np.zeros((64, 64, 3), dtype=np.uint8),
            is_adversarial=True,
            purification_applied=True,
            reconstruction_error=0.1,
            discriminator_score=0.7,
            confidence=0.8,
            details={"test": "value"},
        )

        result_dict = result.to_dict()

        assert result_dict["is_adversarial"] is True
        assert result_dict["purification_applied"] is True
        assert result_dict["reconstruction_error"] == 0.1
        assert result_dict["discriminator_score"] == 0.7
        assert result_dict["confidence"] == 0.8
        assert result_dict["details"]["test"] == "value"

    def test_get_purifier_info(self):
        """Test purifier info retrieval."""
        purifier = DGGANPurifier(latent_dim=64)

        info = purifier.get_purifier_info()

        assert info["latent_dim"] == 64
        assert info["purification_steps"] == 1
        assert info["threshold"] == 0.5
        assert info["encoder_params"] > 0
        assert info["generator_params"] > 0
        assert info["discriminator_params"] > 0

    def test_purify_empty_batch(self):
        """Test batch purification with empty list."""
        purifier = DGGANPurifier(latent_dim=64)

        purified_images, results = purifier.purify_batch([])

        assert len(purified_images) == 0
        assert len(results) == 0

    def test_purify_different_image_sizes(self):
        """Test purification of different image sizes."""
        purifier = DGGANPurifier(latent_dim=64)

        sizes = [(64, 64, 3), (128, 128, 3), (256, 256, 3)]

        for size in sizes:
            image = (np.random.rand(*size) * 255).astype(np.uint8)
            purified, result = purifier.purify(image)

            assert purified.shape == size
            assert isinstance(result, PurifyResult)

    def test_encoder_output_shape(self):
        """Test encoder output shape."""
        purifier = DGGANPurifier(latent_dim=64)

        image = (np.random.rand(64, 64, 3) * 255).astype(np.uint8)
        tensor = purifier._image_to_tensor(image).unsqueeze(0)

        with torch.no_grad():
            latent = purifier.encoder(tensor)

        assert latent.shape[1] == 64
        assert latent.dim() == 4

    def test_generator_output_shape(self):
        """Test generator output shape."""
        purifier = DGGANPurifier(latent_dim=64)

        z = torch.randn(1, 64, 1, 1).to(purifier.device)

        with torch.no_grad():
            output = purifier.generator(z)

        assert output.shape == (1, 3, 64, 64)


class TestPurifyResultConfidence:
    """Test PurifyResult confidence calculations."""

    def test_confidence_range(self):
        """Test that confidence is always in valid range."""
        purifier = DGGANPurifier(latent_dim=64)

        for _ in range(10):
            image = (np.random.rand(64, 64, 3) * 255).astype(np.uint8)
            _, result = purifier.purify(image)

            assert 0 <= result.confidence <= 1

    def test_discriminator_score_range(self):
        """Test that discriminator scores are in valid range."""
        purifier = DGGANPurifier(latent_dim=64)

        for _ in range(10):
            image = (np.random.rand(64, 64, 3) * 255).astype(np.uint8)
            _, result = purifier.purify(image)

            assert 0 <= result.discriminator_score <= 1


class TestDGGANArchitecture:
    """Test DG-GAN architecture components."""

    def test_encoder_parameters(self):
        """Test that encoder has trainable parameters."""
        purifier = DGGANPurifier(latent_dim=64)

        encoder_params = sum(p.numel() for p in purifier.encoder.parameters())
        assert encoder_params > 0

    def test_generator_parameters(self):
        """Test that generator has trainable parameters."""
        purifier = DGGANPurifier(latent_dim=64)

        generator_params = sum(p.numel() for p in purifier.generator.parameters())
        assert generator_params > 0

    def test_discriminator_parameters(self):
        """Test that discriminator has trainable parameters."""
        purifier = DGGANPurifier(latent_dim=64)

        discriminator_params = sum(p.numel() for p in purifier.discriminator.parameters())
        assert discriminator_params > 0

    def test_encoder_output_range(self):
        """Test encoder output range."""
        purifier = DGGANPurifier(latent_dim=64)

        image = (np.random.rand(64, 64, 3) * 255).astype(np.uint8)
        tensor = purifier._image_to_tensor(image).unsqueeze(0)

        with torch.no_grad():
            latent = purifier.encoder(tensor)

        assert not torch.isnan(latent).any()
        assert not torch.isinf(latent).any()

    def test_generator_output_range(self):
        """Test generator output range (should be -1 to 1 due to tanh)."""
        purifier = DGGANPurifier(latent_dim=64)

        z = torch.randn(1, 64, 1, 1).to(purifier.device)

        with torch.no_grad():
            output = purifier.generator(z)

        assert output.min() >= -1.0
        assert output.max() <= 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
