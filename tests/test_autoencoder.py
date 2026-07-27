import torch

from src.models.autoencoder import ConvAutoencoder, reconstruction_errors


def test_autoencoder_preserves_image_shape() -> None:
    model = ConvAutoencoder(latent_channels=16)
    x = torch.rand(2, 3, 64, 64)
    y = model(x)
    assert y.shape == x.shape


def test_reconstruction_errors_one_score_per_image() -> None:
    x = torch.zeros(2, 3, 8, 8)
    y = torch.ones(2, 3, 8, 8)
    scores = reconstruction_errors(x, y)
    assert scores.shape == (2,)
    assert torch.allclose(scores, torch.ones(2))
