"""Regression: checkpoint recomputation must reuse LoRA dropout masks."""
import copy
from types import SimpleNamespace

import torch
import pytest

from lerobot.policies.pi05.lora import LoRALinear
from lerobot.policies.pi05.modeling_pi05 import PI05Pytorch, PaliGemmaWithExpertModel


def test_checkpoint_matches_uncheckpointed_lora_gradients():
    torch.manual_seed(7)
    plain = LoRALinear(torch.nn.Linear(12, 8), rank=4, dropout=0.5)
    # Nonzero B exercises stochastic dependence of both adapter factors.
    torch.nn.init.normal_(plain.lora_B.weight)
    checked = copy.deepcopy(plain)
    x = torch.randn(3, 12)
    state = torch.get_rng_state()
    plain(x).square().mean().backward()
    expected_rng = torch.get_rng_state()
    torch.set_rng_state(state)
    holder = SimpleNamespace(gradient_checkpointing_enabled=True, training=True)
    PI05Pytorch._apply_checkpoint(holder, checked, x).square().mean().backward()
    assert torch.equal(torch.get_rng_state(), expected_rng)
    for (name, p), (_, q) in zip(plain.named_parameters(), checked.named_parameters(), strict=True):
        if p.requires_grad:
            torch.testing.assert_close(p.grad, q.grad, msg=name)


@pytest.mark.parametrize('structured', [False, True])
def test_image_features_tensor_or_structured_output(structured):
    features = torch.randn(1, 4, 12)
    def get_image_features(image):
        assert image.dtype == torch.float32
        return SimpleNamespace(pooler_output=features) if structured else features
    holder = SimpleNamespace(paligemma=SimpleNamespace(model=SimpleNamespace(get_image_features=get_image_features)))
    actual = PaliGemmaWithExpertModel.embed_image(holder, torch.randn(1, 3, 8, 8, dtype=torch.bfloat16))
    torch.testing.assert_close(actual, features.to(torch.bfloat16))
