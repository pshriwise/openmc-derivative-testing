"""Independent transport, visualization, and run controls for the INT launcher."""

from dataclasses import dataclass, field
import json
import math
from pathlib import Path
from .model_defaults import BOX_CONTAINER_LOWER_CM, BOX_CONTAINER_UPPER_CM
from .sampling import AdaptiveSamplingConfig


@dataclass(frozen=True)
class IntRunSettings:
    """Transport, model, output, and sampling settings for INT runs."""

    max_iterations: int = 1
    model_variant: str = 'sphere'
    particles: int = 10000
    adaptive_sampling: AdaptiveSamplingConfig = field(default_factory=AdaptiveSamplingConfig)
    batches: int = 50
    seed: int = 5800001
    threads: int = 2
    paired_sampling: bool = False
    flux_rel_err: float = .01
    trigger_max_factor: int = 1
    trigger_interval: int = 10
    container_radius: float = 13.
    container_margin: float = .01
    box_container_lower: tuple = BOX_CONTAINER_LOWER_CM
    box_container_upper: tuple = BOX_CONTAINER_UPPER_CM
    source_clearance: float = .5
    # The launcher derives this from IntConfig.enable_safeguards for workers.
    validate_geometry: bool = False
    save_geometry_images: bool = True
    save_solid_geometry_images: bool = True
    save_vtk: bool = True
    geometry_plot_pixels: int = 600
    geometry_detail_width: float = 20.
    animation_duration_ms: int = 800
    solid_camera_position: tuple = (24.296, -36.317, -10.742)
    solid_look_at: tuple = (17.723, 4.260, -2.502)
    solid_up: tuple = (.040, -.244, .969)
    solid_pixels: tuple = (1200, 800)
    solid_fov: float = 45.
    box_solid_fov: float = 65.
    solid_diffuse_fraction: float = .7

    def __post_init__(self):
        if isinstance(self.adaptive_sampling, dict):
            object.__setattr__(self, 'adaptive_sampling', AdaptiveSamplingConfig(**self.adaptive_sampling))
        if not isinstance(self.adaptive_sampling, AdaptiveSamplingConfig):
            raise TypeError('adaptive_sampling requires AdaptiveSamplingConfig or its JSON object')
        if self.adaptive_sampling.enabled and self.particles > self.adaptive_sampling.max_particles:
            raise ValueError('Initial particles exceed adaptive_sampling.max_particles')
        if self.model_variant not in ('sphere', 'box'):
            raise ValueError('Unknown INT model variant')
        for name in ('max_iterations', 'particles', 'batches', 'seed', 'threads',
                     'trigger_max_factor', 'trigger_interval', 'geometry_plot_pixels',
                     'animation_duration_ms'):
            value = getattr(self, name)
            minimum = 0 if name == 'max_iterations' else 1
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f'{name} must be an integer >= {minimum}')
        if self.batches < 2:
            raise ValueError('Need at least two transport batches')
        for name in ('paired_sampling', 'save_geometry_images', 'save_solid_geometry_images', 'save_vtk',
                     'validate_geometry'):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f'{name} must be boolean')
        for name in ('flux_rel_err', 'container_radius', 'geometry_detail_width'):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f'{name} must be positive and finite')
        for name in ('container_margin', 'source_clearance'):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f'{name} must be nonnegative and finite')
        if self.container_margin >= self.container_radius:
            raise ValueError('Container margin exceeds its radius')
        for name in ('box_container_lower', 'box_container_upper', 'solid_camera_position',
                     'solid_look_at', 'solid_up'):
            value = getattr(self, name)
            if len(value) != 3 or not all(map(math.isfinite, value)):
                raise ValueError(f'{name} must be a finite XYZ vector')
            object.__setattr__(self, name, tuple(value))
        if any(hi - lo <= 2 * self.container_margin for lo, hi in
               zip(self.box_container_lower, self.box_container_upper)):
            raise ValueError('Box has insufficient room for its margin')
        if (len(self.solid_pixels) != 2 or any(isinstance(v, bool) or not isinstance(v, int)
                                             or v < 1 for v in self.solid_pixels)):
            raise ValueError('solid_pixels must be two positive integers')
        object.__setattr__(self, 'solid_pixels', tuple(self.solid_pixels))
        if (not 0 < self.solid_fov < 180 or not 0 < self.box_solid_fov < 180
                or not 0 <= self.solid_diffuse_fraction <= 1):
            raise ValueError('Invalid solid-rendering settings')
        import numpy as np
        if np.linalg.norm(np.cross(np.subtract(self.solid_look_at, self.solid_camera_position),
                                   self.solid_up)) <= 1e-10:
            raise ValueError('Camera requires a nonzero view and nonparallel up vector')


def load_run_config(path):
    """Require an INT-only run namespace, separate from optimizer parameters."""
    data = json.loads(Path(path).read_text())
    if not isinstance(data, dict) or set(data) != {'int_run'}:
        raise ValueError('Run config must contain only the int_run namespace')
    return IntRunSettings(**data['int_run'])
