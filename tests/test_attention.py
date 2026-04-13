import torch
import torch.nn.functional as F

from model.attention import repeat_kv


def test_repeat_kv_shape() -> None:
    x = torch.randn(2, 2, 5, 8)
    repeated = repeat_kv(x, 4)
    assert repeated.shape == (2, 8, 5, 8)


def test_sdpa_matches_reference_shape() -> None:
    q = torch.randn(1, 2, 4, 8)
    k = torch.randn(1, 2, 4, 8)
    v = torch.randn(1, 2, 4, 8)
    result = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    reference_scores = (q @ k.transpose(-2, -1)) / (8 ** 0.5)
    mask = torch.triu(torch.ones(4, 4, dtype=torch.bool), diagonal=1)
    reference_scores = reference_scores.masked_fill(mask, float("-inf"))
    reference = torch.softmax(reference_scores, dim=-1) @ v
    assert torch.allclose(result, reference, atol=1e-5, rtol=1e-5)
