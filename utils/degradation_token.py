import torch
import torch.nn.functional as F
from torch import nn

from utils.spatial_noise import compute_edge_strength


def _to_luma(image: torch.Tensor) -> torch.Tensor:
    return 0.2989 * image[:, 0:1] + 0.5870 * image[:, 1:2] + 0.1140 * image[:, 2:3]


def _laplacian_energy(gray: torch.Tensor) -> torch.Tensor:
    kernel = torch.tensor(
        [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]],
        device=gray.device,
        dtype=gray.dtype,
    ).view(1, 1, 3, 3)
    return F.conv2d(gray, kernel, padding=1).abs().mean(dim=(1, 2, 3))


def _block_artifact_score(gray: torch.Tensor, block_size: int = 8) -> torch.Tensor:
    height, width = gray.shape[-2:]
    vertical = torch.zeros(gray.shape[0], device=gray.device, dtype=gray.dtype)
    horizontal = torch.zeros_like(vertical)

    if width > block_size:
        boundary_cols = torch.arange(block_size, width, block_size, device=gray.device)
        if boundary_cols.numel() > 0:
            vertical = (gray[:, :, :, boundary_cols] - gray[:, :, :, boundary_cols - 1]).abs().mean(dim=(1, 2, 3))

    if height > block_size:
        boundary_rows = torch.arange(block_size, height, block_size, device=gray.device)
        if boundary_rows.numel() > 0:
            horizontal = (gray[:, :, boundary_rows, :] - gray[:, :, boundary_rows - 1, :]).abs().mean(dim=(1, 2, 3))

    return 0.5 * (vertical + horizontal)


def compute_degradation_stats(lr_rgb: torch.Tensor) -> torch.Tensor:
    """Estimate blur, noise, JPEG, edge, brightness, and contrast in [0, 1] from LR input."""
    if lr_rgb.ndim != 4 or lr_rgb.shape[1] != 3:
        raise ValueError(f"lr_rgb must have shape (B, 3, H, W), got {tuple(lr_rgb.shape)}")

    lr = lr_rgb.float().clamp(0.0, 1.0)
    gray = _to_luma(lr)
    lowpass = F.avg_pool2d(gray, kernel_size=3, stride=1, padding=1)
    highpass = gray - lowpass

    blur = (1.0 - (_laplacian_energy(gray) / 0.20)).clamp(0.0, 1.0)
    noise = (highpass.std(dim=(1, 2, 3)) / 0.12).clamp(0.0, 1.0)
    jpeg = (_block_artifact_score(gray) / 0.10).clamp(0.0, 1.0)
    edge = compute_edge_strength(lr, edge_type="sobel", edge_blur=0).mean(dim=(1, 2, 3)).clamp(0.0, 1.0)
    brightness = gray.mean(dim=(1, 2, 3)).clamp(0.0, 1.0)
    contrast = (gray.std(dim=(1, 2, 3)) / 0.35).clamp(0.0, 1.0)

    return torch.stack([blur, noise, jpeg, edge, brightness, contrast], dim=1)


class DegradationTokenEncoder(nn.Module):
    def __init__(self, stat_dim: int = 6, token_dim: int = 512, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(stat_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, token_dim),
            nn.LayerNorm(token_dim),
        )

    def forward(self, stats: torch.Tensor) -> torch.Tensor:
        return self.net(stats.float()).unsqueeze(1)

    def prepend_to(self, stats: torch.Tensor, image_tokens: torch.Tensor) -> torch.Tensor:
        token = self.forward(stats).to(device=image_tokens.device, dtype=image_tokens.dtype)
        return torch.cat([token, image_tokens], dim=1)
