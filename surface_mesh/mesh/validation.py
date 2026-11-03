"""Geometry checks for exact-coordinate mesh publication; no step correction."""


import numpy as np


from .geometry import quality_metrics, triangle_areas


def volume_tolerance(target, options):
    return options.volume_atol + options.volume_rtol * abs(target)


def check_candidate(reference, candidate, faces, options):
    if not np.isfinite(candidate).all():
        return False, "nonfinite_coordinates"
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        metrics = quality_metrics(reference, candidate, faces)
        minimum = np.maximum(options.absolute_min_area, options.min_area_ratio * triangle_areas(reference, faces))
        if not np.isfinite(metrics["areas"]).all() or np.any(metrics["areas"] < minimum):
            return False, "area_bound"
        if not np.isfinite(metrics["normal_cosine"]).all() or np.any(metrics["normal_cosine"] < options.min_normal_cosine):
            return False, "normal_bound"
        if options.max_vertex_step_fraction is not None:
            shortest = np.full(len(reference), np.inf)
            for a, b in ((0, 1), (1, 2), (2, 0)):
                lengths = np.linalg.norm(reference[faces[:, a]] - reference[faces[:, b]], axis=1)
                np.minimum.at(shortest, faces[:, a], lengths)
                np.minimum.at(shortest, faces[:, b], lengths)
            if np.any(np.linalg.norm(candidate - reference, axis=1) > options.max_vertex_step_fraction * shortest):
                return False, "vertex_step_bound"
    return True, "local_checks_passed"
