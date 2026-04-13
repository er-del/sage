import torch

from model.config import ModelConfig
from model.model import SageTransformer


def test_model_forward_shape_and_weight_tying() -> None:
    config = ModelConfig(
        num_layers=2,
        d_model=64,
        num_attn_heads=4,
        num_kv_heads=2,
        head_dim=16,
        ffn_hidden_dim=256,
        vocab_size=128,
        context_length=32,
    )
    model = SageTransformer(config)
    input_ids = torch.randint(0, config.vocab_size, (2, 8))
    logits, cache = model(input_ids)
    assert logits.shape == (2, 8, config.vocab_size)
    assert len(cache) == config.num_layers
    assert model.embed_tokens.weight.data_ptr() == model.lm_head.weight.data_ptr()
