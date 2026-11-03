"""INT-only sampling policy; never rescales gradients or changes Adam steps."""

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class AdaptiveSamplingConfig:
    enabled: bool = False
    start_iteration: int = 20
    window: int = 3
    patience: int = 2
    cooldown: int = 5
    growth_factor: float = 2.
    max_particles: int = 400000
    objective_relative_range: float = .01
    objective_noise_multiplier: float = 2.
    gradient_relative_threshold: float = .7

    def __post_init__(self):
        if not isinstance(self.enabled, bool):
            raise ValueError('adaptive_sampling.enabled must be boolean')
        for name, minimum in (('start_iteration', 0), ('window', 2), ('patience', 1),
                              ('cooldown', 1), ('max_particles', 1)):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f'adaptive_sampling.{name} must be an integer >= {minimum}')
        for name in ('growth_factor', 'objective_relative_range', 'objective_noise_multiplier',
                     'gradient_relative_threshold'):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or value < 0:
                raise ValueError(f'adaptive_sampling.{name} must be finite and nonnegative')
        if self.growth_factor <= 1 or not 0 <= self.gradient_relative_threshold <= 1:
            raise ValueError('Sampling growth must exceed 1; gradient threshold must be in [0, 1]')


def gradient_norm(sample):
    """L2 norm of gradient per lumped surface area, comparable across remeshes.

    This is only a diagnostic: raw OpenMC derivatives enter the optimizer.
    It does not estimate derivative uncertainty or prove convergence.
    """
    x, f = np.asarray(sample['coordinates']), np.asarray(sample['faces'])
    g = np.asarray(sample['geometry_derivative'])
    thirds = np.linalg.norm(np.cross(x[f[:, 1]] - x[f[:, 0]],
                                    x[f[:, 2]] - x[f[:, 0]]), axis=1) / 6
    area = np.bincount(f.ravel(), weights=np.repeat(thirds, 3), minlength=len(x))
    if g.shape != x.shape or not np.isfinite(g).all() or np.any(area <= 0):
        return None  # Unavailable diagnostic must not become a new geometry gate.
    value = float(np.sqrt(np.sum(np.sum(g * g, axis=1) / area)))
    return value if math.isfinite(value) else None


class AdaptiveSampler:
    """Observe distinct accepted updates; request bounded, persistent increases."""

    def __init__(self, config):
        self.config = config
        self.observations = []
        self.baseline_gradient = None
        self.last_increase = None
        self.streak = 0

    def observe(self, iteration, evaluation, sample):
        norm = gradient_norm(sample)
        row = dict(iteration=iteration, evaluation=evaluation,
                   tally_mean=float(sample['tally_mean']),
                   tally_std_dev=float(sample['tally_std_dev']), gradient_norm=norm)
        if not self.observations:
            self.baseline_gradient = norm
        self.observations.append(row)
        return row

    def decide(self, particles):
        """Call once per new accepted update, before the next optimization step."""
        cfg, latest = self.config, self.observations[-1]
        iteration = latest['iteration']
        result = dict(iteration=iteration, evaluation=latest['evaluation'],
                      particles_before=particles, particles_after=particles,
                      increase=False, reasons=[], status='monitoring')
        if not cfg.enabled or particles >= cfg.max_particles:
            result['status'] = 'disabled' if not cfg.enabled else 'particle_cap'
            return result
        if iteration < cfg.start_iteration or (self.last_increase is not None
                and iteration - self.last_increase < cfg.cooldown):
            self.streak = 0
            result['status'] = 'warmup' if iteration < cfg.start_iteration else 'cooldown'
            return result
        # Exclude pre-increase observations: precision changes start a new window.
        rows = [r for r in self.observations
                if self.last_increase is None or r['iteration'] > self.last_increase][-cfg.window:]
        if len(rows) == cfg.window:
            values = np.array([r['tally_mean'] for r in rows])
            noise = cfg.objective_noise_multiplier * math.sqrt(2) * max(r['tally_std_dev'] for r in rows)
            threshold = max(cfg.objective_relative_range * abs(float(values.mean())), noise)
            span = float(np.ptp(values))
            result.update(objective_range=span, objective_range_threshold=threshold)
            if span <= threshold:
                result['reasons'].append('objective_plateau')
        baseline, norm = self.baseline_gradient, latest['gradient_norm']
        if baseline is not None and norm is not None:
            ratio = norm / baseline if baseline > 0 else (0. if norm == 0 else None)
            result['gradient_ratio_to_initial'] = ratio
            if ratio is not None and ratio <= cfg.gradient_relative_threshold:
                result['reasons'].append('weak_gradient')
        self.streak = self.streak + 1 if result['reasons'] else 0
        result['streak'] = self.streak
        if self.streak >= cfg.patience:
            target = min(cfg.max_particles, math.ceil(particles * cfg.growth_factor))
            result.update(increase=True, particles_after=target, status='increase')
            self.last_increase, self.streak = iteration, 0
        return result
