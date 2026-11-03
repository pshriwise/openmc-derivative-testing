"""Sampling triggers, remesh-independent diagnostics and bounded escalation."""

from dataclasses import replace

import numpy as np
import pytest

from ..core.sampling import AdaptiveSampler, AdaptiveSamplingConfig, gradient_norm
from ..core.settings import IntRunSettings, load_run_config


def sample(mean=1., strength=1., std=.001, refined=False):
    x = np.array([[0., 0., 0.], [1., 0., 0.], [1., 1., 0.], [0., 1., 0.]])
    f = np.array([[0, 1, 2], [0, 2, 3]])
    if refined:
        x = np.vstack((x, [.5, .5, 0.]))
        f = np.array([[0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4]])
    area = np.linalg.norm(np.cross(x[f[:, 1]] - x[f[:, 0]], x[f[:, 2]] - x[f[:, 0]]), axis=1) / 6
    lumped = np.bincount(f.ravel(), weights=np.repeat(area, 3), minlength=len(x))
    g = lumped[:, None] * np.array([[strength, 0., 0.]])
    return dict(coordinates=x, faces=f, geometry_derivative=g, tally_mean=mean, tally_std_dev=std)


def test_area_adjustment_ignores_tessellation_for_same_sensitivity_density():
    assert gradient_norm(sample()) == pytest.approx(1.)
    assert gradient_norm(sample(refined=True)) == pytest.approx(1.)


def test_warmup_persistence_cooldown_and_cap():
    policy = AdaptiveSampler(AdaptiveSamplingConfig(enabled=True, start_iteration=3,
                             patience=2, cooldown=3, max_particles=350, window=3))
    particles = 100
    events = []
    for i in range(12):
        policy.observe(i, i, sample(strength=1 if i == 0 else .5))
        d = policy.decide(particles)
        if d['increase']:
            events.append((i, d['particles_after']))
            particles = d['particles_after']
    assert events == [(4, 200), (8, 350)]
    assert d['status'] == 'particle_cap'


def test_strong_progress_does_not_raise_sampling():
    policy = AdaptiveSampler(AdaptiveSamplingConfig(enabled=True, start_iteration=0))
    for i in range(30):
        policy.observe(i, i, sample(mean=10 - .1 * i))
        assert not policy.decide(50000)['increase']


@pytest.mark.parametrize('sign', [1., -1.])
def test_plateau_is_objective_direction_independent(sign):
    policy = AdaptiveSampler(AdaptiveSamplingConfig(enabled=True, start_iteration=0))
    for i, mean in enumerate([2., 1., 1.002, .999, 1.001]):
        policy.observe(i, i, sample(mean=sign * mean))
        d = policy.decide(50000)
    assert d['increase']
    assert d['reasons'] == ['objective_plateau']


def test_noise_floor_and_zero_initial_gradient():
    cfg = AdaptiveSamplingConfig(enabled=True, start_iteration=0, patience=1)
    policy = AdaptiveSampler(cfg)
    policy.observe(0, 0, sample(strength=0))
    assert policy.decide(50000)['reasons'] == ['weak_gradient']
    policy = AdaptiveSampler(cfg)
    for i, mean in enumerate([1., 1.02, .99]):
        policy.observe(i, i, sample(mean=mean, std=.02))
        d = policy.decide(50000)
    assert d['increase'] and d['reasons'] == ['objective_plateau']


def test_disabled_policy_and_unavailable_geometry_diagnostic():
    p = AdaptiveSampler(AdaptiveSamplingConfig())
    s = sample(); s['coordinates'][:] = 0
    assert gradient_norm(s) is None
    p.observe(0, 0, s)
    assert p.decide(50000)['status'] == 'disabled'


@pytest.mark.parametrize('kwargs', [dict(window=1), dict(patience=0), dict(cooldown=0),
    dict(growth_factor=1), dict(max_particles=0), dict(gradient_relative_threshold=1.1),
    dict(enabled=1), dict(objective_relative_range=float('nan'))])
def test_invalid_sampling_config(kwargs):
    with pytest.raises(ValueError):
        AdaptiveSamplingConfig(**kwargs)


def test_nested_config_roundtrip_and_independent_defaults(tmp_path):
    import json
    from dataclasses import asdict
    cfg = IntRunSettings(adaptive_sampling={'enabled': True, 'start_iteration': 24})
    p = tmp_path / 'run.json'
    p.write_text(json.dumps({'int_run': asdict(cfg)}))
    assert load_run_config(p) == cfg
    assert not IntRunSettings().adaptive_sampling.enabled
    assert replace(cfg, particles=100000).adaptive_sampling == cfg.adaptive_sampling
    with pytest.raises(ValueError, match='exceed'):
        replace(cfg, particles=500000)
