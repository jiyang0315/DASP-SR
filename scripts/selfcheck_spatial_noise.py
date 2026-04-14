import torch
import torch.nn.functional as F

from utils.spatial_noise import compute_edge_strength


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)

    batch_size = 2
    lr_h, lr_w = 128, 128
    lat_h, lat_w = 16, 16

    lr_rgb = torch.rand(batch_size, 3, lr_h, lr_w, device=device)
    latents = torch.randn(batch_size, 4, lat_h, lat_w, device=device)
    noise = torch.randn_like(latents)

    edge_strength = compute_edge_strength(lr_rgb, edge_type="sobel", edge_blur=3)
    assert edge_strength.shape == (batch_size, 1, lr_h, lr_w)
    assert 0.0 <= edge_strength.min().item() <= 1.0
    assert 0.0 <= edge_strength.max().item() <= 1.0

    edge_strength_latent = F.interpolate(edge_strength, size=latents.shape[-2:], mode="bilinear", align_corners=False)
    assert edge_strength_latent.shape == (batch_size, 1, lat_h, lat_w)

    alpha = 0.6
    noise_scale_map = (1.0 - alpha * edge_strength_latent).clamp(min=0.0)
    assert noise_scale_map.shape == (batch_size, 1, lat_h, lat_w)
    assert noise_scale_map.min().item() >= 0.0
    assert noise_scale_map.max().item() <= 1.0 + 1e-6

    sigma = torch.rand(batch_size, 1, 1, 1, device=device)
    sigma_map = sigma * noise_scale_map
    assert sigma_map.shape == (batch_size, 1, lat_h, lat_w)
    assert sigma_map.min().item() >= 0.0
    assert torch.all(sigma_map <= sigma + 1e-6).item()

    alpha0_scale = (1.0 - 0.0 * edge_strength_latent)
    noise_alpha0 = noise * alpha0_scale
    assert torch.allclose(noise_alpha0, noise, atol=0.0, rtol=0.0)

    print(
        "OK",
        {
            "device": device,
            "edge_min": float(edge_strength.min().item()),
            "edge_max": float(edge_strength.max().item()),
            "scale_min": float(noise_scale_map.min().item()),
            "scale_max": float(noise_scale_map.max().item()),
            "sigma_min": float(sigma.min().item()),
            "sigma_max": float(sigma.max().item()),
        },
    )


if __name__ == "__main__":
    main()

