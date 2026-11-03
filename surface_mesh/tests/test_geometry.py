import numpy as np
import pytest

from surface_mesh.mesh.geometry import signed_volume, triangle_areas, triangle_normals, volume_gradient
from surface_mesh.mesh.topology import validate_mesh


def test_tetrahedron_metrics(tetrahedron):
    x, faces = tetrahedron
    assert signed_volume(x, faces) == pytest.approx(1 / 6)
    assert triangle_areas(x, faces).sum() == pytest.approx(1.5 + np.sqrt(3) / 2)
    assert np.allclose(triangle_normals(x, faces).sum(axis=0), 0)
    assert validate_mesh(x, faces)["unused_vertices"] == 0


@pytest.mark.parametrize("scale", [1e-6, 1, 1e6])
def test_scale_and_translation(tetrahedron, scale):
    x, faces = tetrahedron
    x = x * scale + np.array([17, -9, 12]) * scale
    assert signed_volume(x, faces) == pytest.approx(scale**3 / 6)
    gradient = volume_gradient(x, faces)
    assert np.allclose(gradient.sum(axis=0), 0, atol=scale**2 * 1e-14)
    assert validate_mesh(x, faces)["volume"] == pytest.approx(scale**3 / 6)


def test_large_translation(tetrahedron):
    x, faces = tetrahedron
    assert signed_volume(x + 1e9, faces) == pytest.approx(1 / 6)
    assert np.allclose(volume_gradient(x + 1e9, faces), volume_gradient(x, faces))


def test_gradient_matches_finite_difference(tetrahedron):
    x, faces = tetrahedron
    x = x * [2, 0.8, 1.3] + [8, -4, 19]
    gradient = volume_gradient(x, faces)
    origin = x.mean(axis=0)
    epsilon = 1e-5
    for i in range(len(x)):
        for j in range(3):
            direction = np.zeros_like(x)
            direction[i, j] = epsilon
            numerical = (signed_volume(x + direction, faces, origin=origin) - signed_volume(x - direction, faces, origin=origin)) / (2 * epsilon)
            assert numerical == pytest.approx(gradient[i, j], abs=1e-9)


def test_rotation(tetrahedron):
    x, faces = tetrahedron
    angle = 1.2
    rotation = np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
    assert signed_volume(x @ rotation.T, faces) == pytest.approx(signed_volume(x, faces))
    assert np.allclose(triangle_areas(x @ rotation.T, faces), triangle_areas(x, faces))


@pytest.mark.parametrize("case,match", [
    ("open", "Open or nonmanifold"), ("duplicate", "Duplicate"),
    ("winding", "winding"), ("inward", "positive signed volume"),
    ("collapse", "Degenerate"), ("disconnected", "connected"),
])
def test_invalid_boundaries(tetrahedron, case, match):
    x, faces = (a.copy() for a in tetrahedron)
    if case == "open":
        faces = faces[:-1]
    elif case == "duplicate":
        faces = np.vstack((faces, faces[0]))
    elif case == "winding":
        faces[0] = faces[0, ::-1]
    elif case == "inward":
        faces = faces[:, ::-1]
    elif case == "collapse":
        x[3] = x[0]
    elif case == "disconnected":
        x = np.vstack((x, x + 10))
        faces = np.vstack((faces, faces + 4))
    with pytest.raises(ValueError, match=match):
        validate_mesh(x, faces)
