"""Regression tests for cached denoising across Transformers cache APIs."""
import torch
from types import SimpleNamespace

from transformers import DynamicCache, GemmaConfig
from lerobot.policies.common import vla_utils
from lerobot.policies.pi_gemma import PiGemmaModel
from lerobot.policies.pi05.modeling_pi05 import PaliGemmaWithExpertModel


def test_cache_clone_preserves_auxiliary_fields(monkeypatch):
    monkeypatch.setattr(vla_utils, 'DynamicCache', lambda layers: tuple(layers))
    k, v = torch.randn(1, 1, 3, 4), torch.randn(1, 1, 3, 4)
    for layer in ((k, v), (k, v, 128)):
        cloned = vla_utils.clone_past_key_values([layer])[0]
        assert len(cloned) == len(layer)
        assert cloned[2:] == layer[2:]
        assert cloned[0].data_ptr() != k.data_ptr()
        assert cloned[1].data_ptr() != v.data_ptr()
        torch.testing.assert_close(cloned[0], k)
        torch.testing.assert_close(cloned[1], v)


def test_cached_suffix_matches_full_forward_and_leaves_prefix_unchanged():
    torch.manual_seed(17)
    config = GemmaConfig(vocab_size=32, hidden_size=16, intermediate_size=32,
                         num_hidden_layers=2, num_attention_heads=2,
                         num_key_value_heads=1, head_dim=8, pad_token_id=0)
    config._attn_implementation = 'eager'
    model = PiGemmaModel(config).eval()
    x = torch.randn(1, 5, 16)
    mask = torch.zeros(1, 1, 5, 5)
    mask[:, :, :3, 3:] = torch.finfo(torch.float32).min
    with torch.no_grad():
        full = model(inputs_embeds=x, attention_mask=mask, use_cache=False).last_hidden_state
        prefix = model(inputs_embeds=x[:, :3], attention_mask=mask[:, :, :3, :3], use_cache=True)
        cache = prefix.past_key_values
        assert isinstance(cache, DynamicCache)
        original = [(entry[0].clone(), entry[1].clone()) for entry in cache]
        for _ in range(2):
            copied = vla_utils.clone_past_key_values(cache)
            suffix = model(inputs_embeds=x[:, 3:], attention_mask=mask[:, :, 3:, :],
                           past_key_values=copied, use_cache=True).last_hidden_state
            torch.testing.assert_close(suffix, full[:, 3:], atol=1e-5, rtol=1e-5)
            assert copied.get_seq_length() == 5
            assert cache.get_seq_length() == 3
            for entry, expected in zip(cache, original, strict=True):
                torch.testing.assert_close(entry[0], expected[0], atol=0, rtol=0)
                torch.testing.assert_close(entry[1], expected[1], atol=0, rtol=0)


def test_two_stream_training_forward_matches_cached_inference():
    torch.manual_seed(29)
    def model(width, adaptive):
        config = GemmaConfig(vocab_size=32, hidden_size=width, intermediate_size=64,
                             num_hidden_layers=2, num_attention_heads=8,
                             num_key_value_heads=1, head_dim=4, pad_token_id=0,
                             use_adarms=adaptive, adarms_cond_dim=width if adaptive else None)
        config._attn_implementation = 'eager'
        result = PiGemmaModel(config).eval()
        if adaptive:
            for module in result.modules():
                if hasattr(module, 'dense') and module.dense is not None:
                    torch.nn.init.normal_(module.dense.weight, std=0.1)
                    torch.nn.init.normal_(module.dense.bias, std=0.1)
        return result
    prefix, expert = model(32, False), model(24, True)
    holder = SimpleNamespace(training=False,
                             paligemma=SimpleNamespace(model=SimpleNamespace(language_model=prefix)),
                             gemma_expert=SimpleNamespace(model=expert))
    x, y, cond = torch.randn(1, 3, 32), torch.randn(1, 2, 24), torch.randn(1, 24)
    mask = torch.zeros(1, 1, 5, 5)
    mask[:, :, :3, 3:] = torch.finfo(torch.float32).min
    positions = torch.arange(5)[None]
    forward = lambda **kwargs: PaliGemmaWithExpertModel.forward(holder, **kwargs)
    with torch.no_grad():
        joint, _ = forward(inputs_embeds=[x, y], attention_mask=mask,
                           position_ids=positions, adarms_cond=[None, cond], use_cache=False)
        _, cache = forward(inputs_embeds=[x, None], attention_mask=mask[:, :, :3, :3],
                           position_ids=positions[:, :3], use_cache=True)
        cached, _ = forward(inputs_embeds=[None, y], attention_mask=mask[:, :, 3:, :],
                            position_ids=positions[:, 3:], adarms_cond=[None, cond],
                            past_key_values=vla_utils.clone_past_key_values(cache), use_cache=False)
    torch.testing.assert_close(cached[1], joint[1], atol=1e-5, rtol=1e-5)
