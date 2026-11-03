"""NumPy geometry on dense indices, with no dependency on MOAB."""

import math

import numpy as np


def mesh_arrays(coordinates, faces):
    x = np.asarray(coordinates, dtype=np.float64)
    f = np.asarray(faces)
    if x.ndim != 2 or x.shape[1] != 3 or not len(x) or not np.isfinite(x).all():
        raise ValueError("Coordinates must be a nonempty finite (N, 3) array")
    if f.ndim != 2 or f.shape[1] != 3 or not len(f) or f.dtype.kind not in "iu":
        raise ValueError("Faces must be a nonempty integer (M, 3) array")
    if np.any(f < 0) or np.any(f >= len(x)):
        raise ValueError("Face index out of bounds")
    return x, f.astype(np.intp, copy=False)


def triangle_normals(coordinates, faces):
    x, f = mesh_arrays(coordinates, faces)
    return np.cross(x[f[:, 1]] - x[f[:, 0]], x[f[:, 2]] - x[f[:, 0]])


def triangle_areas(coordinates, faces):
    return 0.5 * np.linalg.norm(triangle_normals(coordinates, faces), axis=1)


def reference_point(coordinates):
    x = np.asarray(coordinates)
    return x.min(axis=0) / 2 + x.max(axis=0) / 2


def signed_volume(coordinates, faces, *, origin=None):
    """Signed volume of a closed oriented boundary; no absolute-value repair."""
    x, f = mesh_arrays(coordinates, faces)
    q = x - (reference_point(x) if origin is None else np.asarray(origin))
    terms = np.einsum("ij,ij->i", q[f[:, 0]], np.cross(q[f[:, 1]], q[f[:, 2]]))
    return math.fsum(terms) / 6


def volume_gradient(coordinates, faces, *, origin=None):
    """Gradient for a closed boundary, using a fixed shifted origin."""
    x, f = mesh_arrays(coordinates, faces)
    q = x - (reference_point(x) if origin is None else np.asarray(origin))
    result = np.zeros_like(x)
    for a, b, c in ((0, 1, 2), (1, 2, 0), (2, 0, 1)):
        np.add.at(result, f[:, a], np.cross(q[f[:, b]], q[f[:, c]]) / 6)
    return result


def volume_centroid(coordinates, faces, *, origin=None):
    """Return the enclosed-volume centroid and its (3, N, 3) Jacobian.

    The boundary must be closed and consistently outward oriented. The shifted
    origin is held fixed while differentiating the signed tetrahedron moments.
    """
    x, f = mesh_arrays(coordinates, faces)
    origin = reference_point(x) if origin is None else np.asarray(origin)
    q = x - origin
    a, b, c = (q[f[:, i]] for i in range(3))
    v = np.einsum('ij,ij->i', a, np.cross(b, c)) / 6
    volume = math.fsum(v)
    if volume <= 0 or not np.isfinite(volume):
        raise ValueError('Centroid requires positive enclosed volume')
    centers = (a + b + c) / 4
    relative = np.sum(v[:, None] * centers, axis=0) / volume
    jac = np.zeros((3, len(x), 3))
    for corner, u, w in ((0, b, c), (1, c, a), (2, a, b)):
        dv = np.cross(u, w) / 6
        for k in range(3):
            contribution = (centers[:, k] - relative[k])[:, None] * dv
            contribution[:, k] += v / 4
            np.add.at(jac[k], f[:, corner], contribution / volume)
    return origin + relative, jac


def quality_metrics(reference, candidate, faces):
    original = triangle_normals(reference, faces)
    current = triangle_normals(candidate, faces)
    original_length = np.linalg.norm(original, axis=1)
    current_length = np.linalg.norm(current, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = current_length / original_length
        cosine = np.einsum("ij,ij->i", original, current) / (original_length * current_length)
    return {
        "areas": current_length / 2,
        "area_ratio": ratio,
        "normal_cosine": np.clip(cosine, -1, 1),
    }
