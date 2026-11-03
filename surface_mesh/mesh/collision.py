"""Optional IPC collision queries; no optimization decisions or mesh edits."""

import numpy as np


class CollisionGuard:
    """Check static intersections and linear vertex trajectories with IPC."""

    def __init__(self, coordinates, faces):
        try:
            import ipctk
        except ImportError as exc:
            raise ImportError('Collision checks require the collision extra: pip install ipctk') from exc
        self.ipc = ipctk
        edges = np.unique(np.sort(np.concatenate(
            [faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1), axis=0)
        self.mesh = ipctk.CollisionMesh(coordinates, edges, faces)
        if self.has_intersections(coordinates):
            raise ValueError('Initial mesh has self-intersections')

    def has_intersections(self, coordinates):
        return bool(self.ipc.has_intersections(self.mesh, coordinates))

    def step_limit(self, reference, candidate):
        """Return a conservative collision-free fraction of the linear step."""
        value = float(self.ipc.compute_collision_free_stepsize(
            self.mesh, reference, candidate))
        if not np.isfinite(value) or not 0 <= value <= 1:
            raise ValueError('Collision backend returned an invalid step bound')
        return value


def locally_injective_path(reference, candidate, faces, min_ratio):
    """Require positive projected triangle area throughout a linear step.

    The projected normal is quadratic in time. Checking its endpoints and
    interior minimum detects collapse/folding missed by endpoint normal checks.
    """
    e1 = reference[faces[:, 1]] - reference[faces[:, 0]]
    e2 = reference[faces[:, 2]] - reference[faces[:, 0]]
    d = candidate - reference
    de1 = d[faces[:, 1]] - d[faces[:, 0]]
    de2 = d[faces[:, 2]] - d[faces[:, 0]]
    normal = np.cross(e1, e2)
    norm2 = np.sum(normal * normal, axis=1)
    a = np.sum(np.cross(de1, de2) * normal, axis=1) / norm2
    b = np.sum((np.cross(de1, e2) + np.cross(e1, de2)) * normal, axis=1) / norm2
    minimum = np.minimum(1., 1 + a + b)
    convex = a > 0
    t = np.zeros_like(a)
    t[convex] = np.clip(-b[convex] / (2 * a[convex]), 0, 1)
    minimum = np.minimum(minimum, 1 + b * t + a * t * t)
    return bool(np.isfinite(minimum).all() and np.all(minimum >= min_ratio))
