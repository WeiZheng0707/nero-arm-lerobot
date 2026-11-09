"""Hardware-free checks for environment-state dropout in ACT.

The NERO dataset encodes an undetected cube as all-zero coordinates, so the
policy can learn to treat "zeros" as "no target". Dropout on the whole 14-dim
vector keeps it usable when the detector reports nothing. These tests verify the
dropout applies only while training and never corrupts the forward pass.
"""

import torch

from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACT
from lerobot.configs.types import FeatureType, PolicyFeature


def make_policy(env_dim=14, dropout=0.0):
    config = ACTConfig(
        n_obs_steps=1,
        chunk_size=4,
        n_action_steps=4,
        n_encoder_layers=1,
        n_decoder_layers=1,
        dim_model=32,
        n_heads=2,
        dim_feedforward=64,
        latent_dim=8,
        n_vae_encoder_layers=1,
        env_state_dropout=dropout,
        device="cpu",
    )
    # robot_state_feature / env_state_feature / action_feature are read-only
    # properties derived from these dicts, so they are the only thing to set.
    config.input_features = {
        "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(8,)),
        "observation.environment_state": PolicyFeature(type=FeatureType.ENV, shape=(env_dim,)),
    }
    config.output_features = {
        "action": PolicyFeature(type=FeatureType.ACTION, shape=(8,)),
    }
    return ACT(config)


def make_batch(batch_size=4, env_dim=14):
    return {
        "observation.state": torch.randn(batch_size, 8),
        "observation.environment_state": torch.randn(batch_size, env_dim),
        "action": torch.randn(batch_size, 4, 8),
        "action_is_pad": torch.zeros(batch_size, 4, dtype=torch.bool),
    }


def run_forward(policy, batch):
    """ACT.forward returns (loss, (mu, log_sigma_x2)); only the loss matters here."""
    result = policy.forward(batch)
    return result[0]


def spy_on_env_projection(policy):
    """Record exactly what the env-state projection receives."""
    seen = {}
    original = policy.encoder_env_state_input_proj

    class Spy(torch.nn.Module):
        def forward(self, x):
            seen["x"] = x.detach().clone()
            return original(x)

    policy.encoder_env_state_input_proj = Spy()
    return seen


def test_dropout_defaults_to_off():
    assert ACTConfig().env_state_dropout == 0.0


def test_forward_runs_with_dropout_one():
    # Every sample dropped: the projection still receives a valid zero vector.
    policy = make_policy(dropout=1.0)
    policy.train()
    loss = run_forward(policy, make_batch())
    # `isfinite` returns a tensor; assert on `.all()` so a non-scalar loss still
    # reduces to a single boolean.
    assert torch.isfinite(loss).all()


def test_forward_runs_without_dropout():
    policy = make_policy(dropout=0.0)
    policy.train()
    loss = run_forward(policy, make_batch())
    assert torch.isfinite(loss).all()


def test_dropout_is_inactive_in_eval_mode():
    # Eval must forward the observed vector untouched, otherwise deployment
    # behaviour would silently differ from what the operator sees.
    policy = make_policy(dropout=1.0)
    policy.eval()
    seen = spy_on_env_projection(policy)
    batch = make_batch(batch_size=4)
    with torch.no_grad():
        run_forward(policy, batch)
    assert torch.allclose(seen["x"], batch["observation.environment_state"])


def test_dropout_reaches_the_projection_as_zeros():
    policy = make_policy(dropout=1.0)
    policy.train()
    seen = spy_on_env_projection(policy)
    run_forward(policy, make_batch(batch_size=4))
    assert torch.count_nonzero(seen["x"]) == 0


def test_partial_dropout_keeps_some_rows():
    # With p=0.5 over 32 rows, both kept and dropped rows must appear; seeded so
    # the assertion is deterministic.
    torch.manual_seed(0)
    policy = make_policy(dropout=0.5)
    policy.train()
    seen = spy_on_env_projection(policy)
    run_forward(policy, make_batch(batch_size=32))
    rows_with_signal = int((seen["x"].abs().sum(dim=1) > 0).sum())
    assert 0 < rows_with_signal < 32
