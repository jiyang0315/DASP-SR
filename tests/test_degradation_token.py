import torch

from utils.degradation_token import DegradationTokenEncoder, compute_degradation_stats


def test_compute_degradation_stats_returns_six_normalized_values():
    torch.manual_seed(0)
    clean = torch.linspace(0.0, 1.0, steps=3 * 32 * 32).reshape(1, 3, 32, 32)
    degraded = (clean + 0.05 * torch.randn_like(clean)).clamp(0.0, 1.0)

    stats = compute_degradation_stats(degraded)

    assert stats.shape == (1, 6)
    assert torch.isfinite(stats).all()
    assert torch.all(stats >= 0.0)
    assert torch.all(stats <= 1.0)


def test_degradation_token_encoder_prepends_token_to_image_embeddings():
    torch.manual_seed(0)
    encoder = DegradationTokenEncoder(stat_dim=6, token_dim=512)
    stats = torch.rand(2, 6)
    image_tokens = torch.rand(2, 4, 512)

    combined = encoder.prepend_to(stats, image_tokens)

    assert combined.shape == (2, 5, 512)
    assert torch.allclose(combined[:, 1:], image_tokens)
