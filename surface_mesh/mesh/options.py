"""Checks for mesh publication; these options never modify candidate coordinates."""

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class DeformationOptions:
    min_area_ratio: float = 0.05
    absolute_min_area: float = 0.0
    min_normal_cosine: float = 0.0
    max_vertex_step_fraction: float | None = None
    check_self_intersections: bool = True
    preserve_volume: bool = False
    volume_rtol: float = 1.0e-8
    volume_atol: float = 1.0e-10

    def __post_init__(self):
        for name in ("min_area_ratio", "absolute_min_area", "min_normal_cosine", "volume_rtol", "volume_atol"):
            if not math.isfinite(getattr(self, name)):
                raise ValueError(f"{name} must be finite")
        if not 0 < self.min_area_ratio <= 1 or self.absolute_min_area < 0:
            raise ValueError("Area ratio must be in (0, 1]; absolute area must be nonnegative")
        if not -1 <= self.min_normal_cosine <= 1:
            raise ValueError("Normal cosine must be in [-1, 1]")
        if self.volume_rtol < 0 or self.volume_atol < 0:
            raise ValueError("Volume tolerances must be nonnegative")
        if self.max_vertex_step_fraction is not None:
            if not math.isfinite(self.max_vertex_step_fraction) or self.max_vertex_step_fraction <= 0:
                raise ValueError("Step fraction must be finite and positive")
