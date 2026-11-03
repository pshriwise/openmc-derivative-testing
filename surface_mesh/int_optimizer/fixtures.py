"""Deterministic meshes and derivative fields, independent of transport/MOAB."""

from pathlib import Path

import numpy as np


def sphere(subdivisions=2, radius=1.0):
    """Return an outward triangular sphere by subdividing an octahedron."""
    if isinstance(subdivisions, bool) or not isinstance(subdivisions, int) or subdivisions < 0:
        raise ValueError('subdivisions must be a nonnegative integer')
    if not np.isfinite(radius) or radius <= 0:
        raise ValueError('radius must be positive and finite')
    vertices = np.array([[1., 0, 0], [-1., 0, 0], [0, 1., 0],
                         [0, -1., 0], [0, 0, 1.], [0, 0, -1.]])
    faces = np.array([[0, 2, 4], [2, 1, 4], [1, 3, 4], [3, 0, 4],
                      [2, 0, 5], [1, 2, 5], [3, 1, 5], [0, 3, 5]])
    for _ in range(subdivisions):
        points, cache, triangles = list(vertices), {}, []
        def midpoint(a, b):
            key = tuple(sorted((a, b)))
            if key not in cache:
                point = vertices[a] + vertices[b]
                cache[key] = len(points)
                points.append(point / np.linalg.norm(point))
            return cache[key]
        for a, b, c in faces:
            ab, bc, ca = midpoint(a, b), midpoint(b, c), midpoint(c, a)
            triangles.extend([[a, ab, ca], [ab, b, bc], [ca, bc, c], [ab, bc, ca]])
        vertices, faces = np.asarray(points), np.asarray(triangles)
    return vertices * radius, faces


def gradient_field(vertices, case):
    """Return raw synthetic dT/dv; no automatic magnitude normalization."""
    gradient = np.zeros_like(vertices)
    if case == 'localized':
        gradient[np.argmax(vertices[:, 0]), 0] = 1.
    elif case == 'smooth':
        gradient = vertices * [1., 0.2, -0.4]
    elif case == 'radial':
        gradient = vertices.copy()
    elif case != 'zero':
        raise ValueError(f'Unknown gradient case: {case}')
    return gradient


def write_obj(path, vertices, faces):
    """Export fixed triangle connectivity for inspection; never overwrite."""
    with Path(path).open('x') as stream:
        for point in vertices:
            stream.write('v ' + ' '.join(format(x, '.17g') for x in point) + '\n')
        for face in faces:
            stream.write('f ' + ' '.join(str(int(i) + 1) for i in face) + '\n')
