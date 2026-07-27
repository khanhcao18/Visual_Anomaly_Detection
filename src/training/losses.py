import torch


def mse_reconstruction_loss() -> torch.nn.MSELoss:
    return torch.nn.MSELoss()
