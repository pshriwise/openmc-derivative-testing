"""Settings exclusively for the INT-inspired experiment, never legacy options."""

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class IntConfig:
    """Configure a frozen-gradient inner solve in inherited coordinate units.

    Volume weight multiplies absolute volume error squared (length**6).
    Bounds are three optional (lower, upper) pairs, in the mesh-local frame.
    No derivative normalization, equality projection, or centroid fixing occurs.
    Collision checks require the optional IPC Toolkit dependency.
    Added geometry/numerical acceptance gates are inactive unless
    enable_safeguards is true. Finite-array and file/mapping contracts remain.
    The transport launcher interprets remesh_interval in outer updates (0 off),
    with projected Botsch iterations and an optional target edge length.
    persist_adam_state carries Adam moments and step count across outer
    updates. Its momentum only damps outer-update oscillation when iterations
    per gradient is small (INT itself takes one Adam step per gradient).
    remesh_adam_state 'transfer' maps moments by nearest vertex after a remesh;
    'reset' starts a fresh optimizer, as INT does. A positive
    remesh_learning_rate_per_edge then sets the learning rate to that factor
    times the remesh target edge length (INT: 0.07). remesh_halve_edge_rounds
    lists 1-based remesh counts whose target edge length is halved (INT: 3, 5).
    """

    large_steps_lambda: float = 35.0
    learning_rate: float = 0.005
    iterations: int = 20
    volume_weight: float = 5.0
    range_weight: float = 2.0
    bounds: tuple = (None, None, None)
    objective: str = 'minimize'
    max_displacement: float = 0.1
    min_area_ratio: float = 0.05
    min_normal_cosine: float = 0.0
    min_edge_ratio: float = 1e-6
    check_self_intersections: bool = False
    backend: str = 'auto'
    enable_safeguards: bool = False
    remesh_interval: int = 0
    remesh_iterations: int = 5
    remesh_target_edge_length: float | None = None
    persist_adam_state: bool = False
    remesh_adam_state: str = 'transfer'
    remesh_learning_rate_per_edge: float | None = None
    remesh_halve_edge_rounds: tuple = ()

    def __post_init__(self):
        for name in ('enable_safeguards', 'persist_adam_state'):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f'{name} must be boolean')
        if self.remesh_adam_state not in ('transfer', 'reset'):
            raise ValueError('remesh_adam_state must be transfer or reset')
        rate = self.remesh_learning_rate_per_edge
        if rate is not None and (not math.isfinite(rate) or rate <= 0):
            raise ValueError('remesh_learning_rate_per_edge must be null or positive and finite')
        rounds = tuple(self.remesh_halve_edge_rounds)
        if any(isinstance(r, bool) or not isinstance(r, int) or r < 1 for r in rounds):
            raise ValueError('remesh_halve_edge_rounds must contain integers >= 1')
        object.__setattr__(self, 'remesh_halve_edge_rounds', rounds)
        for name, minimum in (('remesh_interval', 0), ('remesh_iterations', 1)):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f'{name} must be an integer >= {minimum}')
        if self.remesh_target_edge_length is not None and (
                not math.isfinite(self.remesh_target_edge_length)
                or self.remesh_target_edge_length <= 0):
            raise ValueError('remesh_target_edge_length must be null or positive and finite')
        for name in ('large_steps_lambda', 'volume_weight', 'range_weight'):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f'{name} must be finite and nonnegative')
        for name in ('learning_rate', 'max_displacement'):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f'{name} must be finite and positive')
        if (isinstance(self.iterations, bool) or not isinstance(self.iterations, int)
                or self.iterations < 0):
            raise ValueError('iterations must be a nonnegative integer')
        if self.objective not in ('minimize', 'maximize'):
            raise ValueError('objective must be minimize or maximize')
        if self.backend not in ('auto', 'llvm_ad_rgb', 'cuda_ad_rgb'):
            raise ValueError('backend must be auto, llvm_ad_rgb, or cuda_ad_rgb')
        if not isinstance(self.check_self_intersections, bool):
            raise ValueError('check_self_intersections must be boolean')
        if (not 0 < self.min_area_ratio <= 1 or not 0 < self.min_edge_ratio <= 1
                or not -1 <= self.min_normal_cosine <= 1):
            raise ValueError('Invalid quality threshold')
        if len(self.bounds) != 3:
            raise ValueError('bounds must contain three axis entries')
        bounds = []
        for bound in self.bounds:
            if bound is not None and (len(bound) != 2 or not all(map(math.isfinite, bound))
                                      or bound[0] >= bound[1]):
                raise ValueError('Each bound must be null or a finite increasing pair')
            bounds.append(None if bound is None else tuple(bound))
        object.__setattr__(self, 'bounds', tuple(bounds))


def load_config(path):
    """Load the isolated JSON namespace; reject legacy or misspelled settings."""
    import json
    from pathlib import Path
    data = json.loads(Path(path).read_text())
    if not isinstance(data, dict) or set(data) != {'int_inspired'}:
        raise ValueError('Config must contain only the int_inspired namespace')
    return IntConfig(**data['int_inspired'])
