"""Constraint API checks, including collisions missed by endpoint-only checks."""

import numpy as np
import pytest

from surface_mesh.mesh.geometry import volume_centroid
from surface_mesh.mesh.collision import CollisionGuard, locally_injective_path


@pytest.fixture
def exclusion_mesh(tetrahedron):
    from pathlib import Path
    from surface_mesh.mesh import MeshSnapshot
    x, f = tetrahedron
    return MeshSnapshot(Path('unused.h5m'), '', 0.1 * x + [1, 0, 0], f,
                        np.arange(len(x)), 1, 1)


def exclusion_checks(mesh, **kwargs):
    from surface_mesh.mesh import MeshConstraints, DeformationOptions
    return MeshConstraints(mesh, options=DeformationOptions(check_self_intersections=False),
                           **kwargs)


def test_vertex_exclusion_endpoints_and_path(exclusion_mesh):
    mesh = exclusion_mesh
    checks = exclusion_checks(mesh, vertex_exclusion_center=[0, 0, 0],
                              vertex_exclusion_radius=0.5)
    # Exactly on the boundary is allowed, including stationary vertices.
    metrics = checks.validate(mesh.coordinates - [0.5, 0, 0])
    assert metrics['min_vertex_exclusion_distance'] == 0.5
    assert metrics['min_vertex_exclusion_path_distance'] == 0.5
    checks.validate(mesh.coordinates)
    with pytest.raises(ValueError, match='Vertex exclusion violated'):
        checks.validate(mesh.coordinates - [0.51, 0, 0])
    # Both endpoints are safe, but a rigid translation crosses the source.
    candidate = mesh.coordinates - [2, 0, 0]
    assert np.min(np.linalg.norm(candidate, axis=1)) > 0.5
    with pytest.raises(ValueError, match='along its linear update'):
        checks.validate(candidate)
    exclusion_checks(mesh).validate(candidate)  # Optional, backward-compatible API.


def test_vertex_exclusion_rejects_initial_mesh(exclusion_mesh):
    with pytest.raises(ValueError, match='Vertex exclusion violated'):
        exclusion_checks(exclusion_mesh, vertex_exclusion_center=[1, 0, 0],
                         vertex_exclusion_radius=0.5)


def test_vertex_exclusion_rejects_before_writing(exclusion_mesh, tmp_path):
    checks = exclusion_checks(exclusion_mesh, vertex_exclusion_center=[0, 0, 0],
                              vertex_exclusion_radius=0.5)
    output = tmp_path / 'invalid.h5m'
    with pytest.raises(ValueError, match='Vertex exclusion violated'):
        checks.write(exclusion_mesh.coordinates - [0.75, 0, 0], output)
    assert not output.exists()


@pytest.mark.parametrize('kwargs', [
    {'vertex_exclusion_radius': -1}, {'vertex_exclusion_radius': np.nan},
    {'vertex_exclusion_radius': np.inf}, {'vertex_exclusion_radius': 0.5},
    {'vertex_exclusion_center': [0, 0]}, {'vertex_exclusion_center': [0, np.nan, 0]},
])
def test_invalid_vertex_exclusion(exclusion_mesh, kwargs):
    with pytest.raises(ValueError, match='exclusion'):
        exclusion_checks(exclusion_mesh, **kwargs)


def test_centroid_jacobian(tetrahedron):
    x, f = tetrahedron
    x = x + [100., -30., 20.]
    center, jac = volume_centroid(x, f)
    np.testing.assert_allclose(center, x.mean(axis=0))
    for i in range(len(x)):
        for k in range(3):
            delta = np.zeros_like(x)
            delta[i, k] = 1e-5
            fd = (volume_centroid(x + delta, f)[0] - volume_centroid(x - delta, f)[0]) / 2e-5
            np.testing.assert_allclose(jac[:, i, k], fd, atol=2e-9)


def test_crossing_without_normal_change():
    pytest.importorskip('ipctk')
    x = np.array([[0., 0, 0], [1, 0, 0], [0, 1, 0],
                  [0., 0, 1], [1, 0, 1], [0, 1, 1]])
    f = np.array([[0, 1, 2], [3, 4, 5]])
    y = x.copy()
    y[3:, 2] = -1
    guard = CollisionGuard(x, f)
    assert not guard.has_intersections(y)
    assert locally_injective_path(x, y, f, 0.1)
    assert 0 < guard.step_limit(x, y) < 0.5


def test_collapse_and_recovery():
    x = np.array([[0., 0, 0], [1, 0, 0], [0, 1, 0]])
    y = -x
    f = np.array([[0, 1, 2]])
    # Normals are identical at the endpoints; the triangle collapses at t=1/2.
    assert not locally_injective_path(x, y, f, 0.1)


@pytest.mark.integration
def test_exact_writer_and_legacy_material_group(tmp_path):
    pytest.importorskip('pymoab')
    pytest.importorskip('pydagmc')
    pytest.importorskip('ipctk')
    from surface_mesh.mesh import load_mesh, MeshConstraints, DeformationOptions
    from surface_mesh.mesh.moab_adapter import MoabMesh
    from surface_mesh.mesh.reporting import file_hash
    from pathlib import Path
    fixture = Path(__file__).resolve().parents[1] / 'sphere.h5m'
    source = tmp_path / 'source.h5m'
    source.write_bytes(fixture.read_bytes())
    m = MoabMesh(source)
    model = m.dagmc_model()
    # Mimic the valid legacy group in the user's model, without a dimension tag.
    group = model.groups[0]
    m.mb.tag_set_data(m.mb.tag_get_handle('GEOM_DIMENSION'), [group.handle], [-1])
    legacy = tmp_path / 'legacy.h5m'
    m.write(legacy)
    digest = file_hash(legacy)
    s = load_mesh(legacy)
    center, _ = volume_centroid(s.coordinates, s.faces)
    checks = MeshConstraints(s, options=DeformationOptions(preserve_volume=True), centroid_target=center)
    y = (s.coordinates - center) * [1.1, 1 / 1.1, 1] + center
    output = tmp_path / 'accepted.h5m'
    metrics, actual = checks.write(y, output)
    np.testing.assert_array_equal(actual, y - s.coordinates)
    assert file_hash(legacy) == digest
    assert abs(metrics['volume_residual']) < 1e-7
    assert load_mesh(output).surface_id == s.surface_id
    with pytest.raises(ValueError, match='Volume'):
        checks.write(s.coordinates * 1.1, tmp_path / 'invalid.h5m')
    assert not (tmp_path / 'invalid.h5m').exists()
