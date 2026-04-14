import torch
import torch.nn.functional as F


def compute_edge_strength(lr_rgb: torch.Tensor, edge_type: str = "sobel", edge_blur: int = 0) -> torch.Tensor:
    """
    Compute a normalized edge strength map in [0, 1].

    Args:
        lr_rgb: (B, 3, H, W) tensor, expected in [0, 1] (but not strictly required).
        edge_type: "sobel" or "laplacian".
        edge_blur: Optional blur kernel size (0 disables). If even, it will be rounded up to the next odd value.

    Returns:
        edge_strength: (B, 1, H, W) tensor in [0, 1].
    """
    if lr_rgb.ndim != 4 or lr_rgb.shape[1] != 3:
        raise ValueError(f"lr_rgb must have shape (B, 3, H, W), got {tuple(lr_rgb.shape)}")

    x = lr_rgb
    if not x.is_floating_point():
        x = x.float()

    # Convert to grayscale for edge detection
    gray = 0.2989 * x[:, 0:1, ...] + 0.5870 * x[:, 1:2, ...] + 0.1140 * x[:, 2:3, ...]

    edge_type = (edge_type or "").lower()
    if edge_type == "sobel":
        kx = torch.tensor(
            [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
            device=gray.device,
            dtype=gray.dtype,
        ).view(1, 1, 3, 3)
        ky = torch.tensor(
            [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]],
            device=gray.device,
            dtype=gray.dtype,
        ).view(1, 1, 3, 3)
        gx = F.conv2d(gray, kx, padding=1)
        gy = F.conv2d(gray, ky, padding=1)
        edge = torch.sqrt(gx * gx + gy * gy + 1e-12)
    elif edge_type == "laplacian":
        k = torch.tensor(
            [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]],
            device=gray.device,
            dtype=gray.dtype,
        ).view(1, 1, 3, 3)
        edge = F.conv2d(gray, k, padding=1).abs()
    else:
        raise ValueError(f"Unknown edge_type={edge_type!r}, expected 'sobel' or 'laplacian'")

    if edge_blur and int(edge_blur) > 0:
        k = int(edge_blur)
        if k % 2 == 0:
            k += 1
        edge = F.avg_pool2d(edge, kernel_size=k, stride=1, padding=k // 2)

    # Per-image normalize to [0, 1]
    edge_min = edge.amin(dim=(2, 3), keepdim=True)
    edge_max = edge.amax(dim=(2, 3), keepdim=True)
    edge = (edge - edge_min) / (edge_max - edge_min + 1e-12)
    return edge.clamp(0.0, 1.0)

