"""Numerical acceptance tests with actual Mitsuba/Dr.Jit, no transport."""

from dataclasses import replace
import json

import numpy as np
import pytest

pytest.importorskip('mitsuba')
pytest.importorskip('cholespy')

from surface_mesh.mesh.geometry import signed_volume, volume_gradient
from surface_mesh.int_optimizer import IntConfig, optimize_vertices
from surface_mesh.int_optimizer.backend import initialize_backend, loss_terms
from surface_mesh.int_optimizer.config import load_config
from surface_mesh.int_optimizer.engine import geometry_hash
from surface_mesh.int_optimizer.fixtures import sphere, gradient_field
from surface_mesh.int_optimizer.interchange import align_gradient_ids, load_derivatives


@pytest.fixture
def mesh():
    return sphere(1)


def latent(mesh):
    x, f = mesh
    mi, dr, _ = initialize_backend('llvm_ad_rgb')
    ls = mi.ad.LargeSteps(mi.Float(x.ravel()), mi.UInt(f.ravel()), 35.)
    u = dr.detach(ls.to_differential(mi.Float(x.ravel())))
    dr.enable_grad(u)
    return mi, dr, ls, u


def test_largesteps_roundtrip(mesh):
    x, f = mesh
    mi, dr, ls, u = latent(mesh)
    actual = ls.from_differential(u).numpy().reshape(-1, 3)
    np.testing.assert_allclose(actual, x, rtol=0, atol=2e-6)
    # Conversion preserves XYZ order despite LargeSteps' internal unique sorting.
    np.testing.assert_allclose(dr.unravel(mi.Point3f, mi.Float(x.ravel())).numpy().T, x, atol=1e-7)


def test_external_gradient_matches_autodiff(mesh):
    x, f = mesh
    mi, dr, ls, u = latent(mesh)
    weights = np.linspace(.1, 2., x.size)
    v = ls.from_differential(u)
    reference = v.numpy().copy()
    dr.backward(.5 * dr.sum(mi.Float(weights) * dr.square(v)))
    expected = dr.grad(u).numpy().copy()
    u2 = dr.detach(mi.Float(u))
    dr.enable_grad(u2)
    v2 = ls.from_differential(u2)
    gradient = (weights * reference).reshape(-1, 3)
    terms = loss_terms(mi, dr, v2, mi.Float(reference), gradient, f,
                       IntConfig(volume_weight=0, range_weight=0), signed_volume(x, f), [0, 0, 0])
    dr.backward(terms[0])
    np.testing.assert_allclose(dr.grad(u2).numpy(), expected, rtol=2e-5, atol=1e-7)


def test_external_gradient_finite_difference(mesh):
    x, f = mesh
    mi, dr, ls, u = latent(mesh)
    g = np.random.default_rng(72).normal(size=x.shape)
    dr.backward(dr.sum(mi.Float(g.ravel()) * ls.from_differential(u)))
    analytic = dr.grad(u).numpy()
    direction = np.random.default_rng(29).normal(size=x.size)
    direction /= np.linalg.norm(direction)
    values = []
    epsilon = .05  # Resolves float32 sparse-solve noise in latent coordinates.
    for sign in (-1, 1):
        v = ls.from_differential(mi.Float(u.numpy() + sign * epsilon * direction)).numpy()
        values.append(float(g.ravel() @ v))
    assert (values[1] - values[0]) / (2 * epsilon) == pytest.approx(analytic @ direction, rel=.002, abs=2e-5)


def test_combined_penalties_keep_transport_adjoint(mesh):
    # Regression for mixing a flat transport branch and unraveled volume branch:
    # Dr.Jit 1.5.0 can overwrite the transport adjoint at a permuting gather.
    x, f = mesh
    mi, dr, ls, u = latent(mesh)
    v = ls.from_differential(u)
    current = v.numpy().reshape(-1, 3).astype(float)
    g = gradient_field(x, 'smooth')
    cfg = IntConfig(bounds=((-0.8, .7), None, None), volume_weight=5., range_weight=2.)
    target = signed_volume(x, f) * .9
    terms = loss_terms(mi, dr, v, mi.Float(x.ravel()), g, f, cfg, target, [0, 0, 0])
    dr.backward(terms[0])
    actual = dr.grad(u).numpy().copy()
    total = g + 2 * cfg.volume_weight * (signed_volume(current, f) - target) * volume_gradient(current, f)
    incidence = np.bincount(f.ravel(), minlength=len(x))
    total[:, 0] += 2 * cfg.range_weight * incidence * (
        np.minimum(current[:, 0] + .8, 0) + np.maximum(current[:, 0] - .7, 0))
    u2 = dr.detach(mi.Float(u))
    dr.enable_grad(u2)
    dr.backward(dr.sum(mi.Float(total.ravel()) * ls.from_differential(u2)))
    np.testing.assert_allclose(actual, dr.grad(u2).numpy(), rtol=2e-4, atol=3e-6)


def test_zero_gradient_and_final_post_update(mesh):
    x, f = mesh
    zero = optimize_vertices(x, f, np.zeros_like(x))
    np.testing.assert_array_equal(zero.vertices, x)
    assert zero.status == 'stationary'
    g = gradient_field(x, 'localized')
    for objective, sign in [('minimize', -1), ('maximize', 1)]:
        result = optimize_vertices(x, f, g, IntConfig(iterations=1, volume_weight=0, objective=objective))
        assert sign * np.sum(g * (result.vertices - x)) > 0.004
        np.testing.assert_array_equal(result.faces, f)
        assert result.history[-1]['predicted_tally_change'] == pytest.approx(np.sum(g * (result.vertices - x)))


def test_volume_signed_and_sphere_reference():
    x, f = sphere(4, 5.)
    assert signed_volume(x, f) == pytest.approx(4/3 * np.pi * 125, rel=.006)
    assert signed_volume(x, f[:, ::-1]) < 0


def test_int_divergence_volume_matches_reference(mesh):
    x, f = mesh
    x = x * [1.2, .8, 1.1] + [2., -3., 1.]
    a, b, c = (x[f[:, k]] for k in range(3))
    component_formula = np.sum((a[:, 1] + b[:, 1] + c[:, 1]) * np.cross(b - a, c - a)[:, 1]) / 6
    assert component_formula == pytest.approx(signed_volume(x, f), rel=1e-14)
    mi, dr, _ = initialize_backend()
    terms = loss_terms(mi, dr, mi.Float(x.ravel()), mi.Float(x.ravel()), np.zeros_like(x),
                       f, IntConfig(), signed_volume(x, f), [2, -3, 1])
    assert float(terms[4][0]) == pytest.approx(signed_volume(x, f), rel=1e-6)


def test_volume_penalty_reduces_volume_error():
    x, f = sphere(2)
    cfg = IntConfig(iterations=60)
    free = optimize_vertices(x, f, x, replace(cfg, volume_weight=0))
    penalized = optimize_vertices(x, f, x, cfg)
    assert free.status == penalized.status == 'completed'
    assert abs(penalized.diagnostics['final']['volume_change']) < abs(free.diagnostics['final']['volume_change'])


def test_range_penalty_counts_incident_corners(mesh):
    x, f = mesh
    mi, dr, ls, u = latent(mesh)
    cfg = IntConfig(bounds=((-0.5, 0.5), None, None), volume_weight=0)
    terms = loss_terms(mi, dr, mi.Float(x.ravel()), mi.Float(x.ravel()), np.zeros_like(x),
                       f, cfg, signed_volume(x, f), [0, 0, 0])
    expected = np.sum(np.maximum(np.abs(x[f.ravel(), 0]) - .5, 0)**2)
    assert float(terms[3][0]) == pytest.approx(expected, rel=1e-6)
    result = optimize_vertices(x, f, np.zeros_like(x), replace(cfg, iterations=1))
    assert result.history[0]['range_penalty'] < expected


def test_reject_invalid_mesh_and_trust_region(mesh):
    x, f = mesh
    with pytest.raises(ValueError, match='Open or nonmanifold'):
        optimize_vertices(x, f[:-1], x, IntConfig(enable_safeguards=True))
    with pytest.raises(ValueError, match='outward winding'):
        optimize_vertices(x, f[:, ::-1], x, IntConfig(enable_safeguards=True))
    with pytest.raises(ValueError, match='finite'):
        optimize_vertices(x, f, x * np.nan)
    result = optimize_vertices(x, f, gradient_field(x, 'localized'),
                               IntConfig(max_displacement=1e-5, volume_weight=0, enable_safeguards=True))
    assert result.status == 'rejected_step'
    assert not result.history[-1]['accepted']
    np.testing.assert_array_equal(result.vertices, x)


def test_separate_config_namespace(tmp_path, mesh):
    path = tmp_path / 'config.json'
    path.write_text(json.dumps({'int_inspired': {'iterations': 1}}))
    assert load_config(path).large_steps_lambda == 35
    path.write_text(json.dumps({'smoothing': 1}))
    with pytest.raises(ValueError, match='namespace'):
        load_config(path)
    path.write_text(json.dumps({'int_inspired': {'smoothing': 1}}))
    with pytest.raises(TypeError):
        load_config(path)
    from surface_mesh.mesh import DeformationOptions
    with pytest.raises(TypeError, match='IntConfig'):
        optimize_vertices(*mesh, mesh[0], DeformationOptions())


def test_gradient_index_mapping():
    np.testing.assert_array_equal(align_gradient_ids([30, 10, 20],
        [[3., 0, 0], [1, 0, 0], [2, 0, 0]], [10, 20, 30])[:, 0], [1, 2, 3])
    for ids in ([10, 10, 30], [10, 20, 40], [10, 20]):
        with pytest.raises(ValueError):
            align_gradient_ids(ids, np.ones((3, 3)), [10, 20, 30])


def test_bundle_staleness(mesh, tmp_path):
    x, f = mesh
    metadata = dict(tally_id='synthetic', objective='minimize', coordinate_units='cm',
                    derivative_units='response/cm', geometry_sha256=geometry_hash(x, f))
    path = tmp_path / 'derivative.npz'
    data = dict(vertices=x, faces=f, gradients=x, vertex_ids=np.arange(len(x)), metadata=json.dumps(metadata))
    np.savez(path, **data)
    bundle = load_derivatives(path)
    np.testing.assert_array_equal(bundle.gradients, x)
    np.savez(path, **dict(data, vertices=x * 1.001))
    with pytest.raises(ValueError, match='Stale'):
        load_derivatives(path)


def test_h5m_roundtrip_preserves_metadata(tmp_path, sphere_path):
    pytest.importorskip('pymoab')
    from surface_mesh.int_optimizer.interchange import load_mesh, write_displaced_h5m
    from surface_mesh.mesh.moab_adapter import MoabMesh
    from surface_mesh.mesh.reporting import file_hash
    source = sphere_path.with_suffix('.h5m')
    snapshot = load_mesh(source)
    original = file_hash(source)
    candidate = snapshot.coordinates + [0.001, -.002, .001]
    cfg = IntConfig()
    output = tmp_path / 'updated.h5m'
    write_displaced_h5m(snapshot, output, candidate, cfg, dry_run=True)
    assert not output.exists()
    write_displaced_h5m(snapshot, output, candidate, cfg)
    before, after = MoabMesh(source), MoabMesh(output)
    assert before.inventory() == after.inventory()
    np.testing.assert_array_equal(before.vertex_handles, after.vertex_handles)
    np.testing.assert_array_equal(before.triangle_handles, after.triangle_handles)
    np.testing.assert_array_equal(before.faces, after.faces)
    np.testing.assert_array_equal(after.coordinates, candidate)
    assert file_hash(source) == original
    with pytest.raises((ValueError, FileExistsError)):
        write_displaced_h5m(snapshot, source, candidate, cfg)


def test_genesis_adapter_rejects_stale_handles(tmp_path, sphere_path):
    pytest.importorskip('pymoab')
    from surface_mesh.int_optimizer.interchange import load_mesh
    mesh = load_mesh(sphere_path.with_suffix('.h5m'))
    data = dict(coordinates=mesh.coordinates, faces=mesh.faces, vertex_handles=mesh.vertex_handles,
                geometry_derivative=mesh.coordinates, input_sha256=mesh.input_sha256)
    metadata = dict(tally_id=1, coordinate_units='cm', derivative_units='response/cm', objective='minimize')
    path = tmp_path / 'sample.npz'
    np.savez(path, **data)
    result = load_derivatives(path, snapshot=mesh, legacy_metadata=metadata)
    np.testing.assert_array_equal(result.gradients, mesh.coordinates)
    order = np.arange(len(mesh.coordinates))[::-1]
    inverse = np.empty_like(order)
    inverse[order] = np.arange(len(order))
    np.savez(path, **dict(data, coordinates=mesh.coordinates[order],
                         geometry_derivative=mesh.coordinates[order],
                         vertex_handles=mesh.vertex_handles[order], faces=inverse[mesh.faces]))
    reordered = load_derivatives(path, snapshot=mesh, legacy_metadata=metadata)
    np.testing.assert_array_equal(reordered.gradients, mesh.coordinates)
    np.savez(path, **dict(data, input_sha256='stale'))
    with pytest.raises(ValueError, match='file hash'):
        load_derivatives(path, snapshot=mesh, legacy_metadata=metadata)
    np.savez(path, **dict(data, vertex_handles=mesh.vertex_handles[::-1]))
    with pytest.raises(ValueError, match='coordinates/connectivity'):
        load_derivatives(path, snapshot=mesh, legacy_metadata=metadata)


def test_cli_demo(tmp_path):
    from surface_mesh.cli.deform import main
    cfg = tmp_path / 'config.json'
    cfg.write_text('{"int_inspired": {"iterations": 1}}')
    output = tmp_path / 'demo'
    assert main(['--int-config', str(cfg), '--demo', 'localized', '--output-dir', str(output)]) == 0
    report = json.loads((output / 'history.json').read_text())
    assert report['method'] == 'int_inspired'
    assert (output / 'optimized.obj').exists()
    with pytest.raises(FileExistsError):
        main(['--int-config', str(cfg), '--demo', 'localized', '--output-dir', str(output)])


def test_disabled_safeguards_ignore_quality_and_trust_limits(mesh, tmp_path, sphere_path):
    from surface_mesh.int_optimizer.engine import validate_candidate
    from surface_mesh.int_optimizer.interchange import load_mesh, write_displaced_h5m
    x, f = mesh
    cfg = IntConfig(max_displacement=1e-12, min_area_ratio=1., min_normal_cosine=1.,
                    min_edge_ratio=1., check_self_intersections=True, iterations=1)
    # Every geometric restriction is subordinate to the disabled master switch.
    validate_candidate(x, .5 * x, f, cfg)
    result = optimize_vertices(x, f, gradient_field(x, 'localized'), cfg)
    assert result.status == 'completed'
    assert result.diagnostics['self_intersections'] == 'unverified'
    source = load_mesh(sphere_path.with_suffix('.h5m'))
    write_displaced_h5m(source, tmp_path / 'unchecked.h5m', .5 * source.coordinates, cfg)
    with pytest.raises(ValueError, match='Quality check'):
        validate_candidate(x, .5 * x, f, replace(cfg, enable_safeguards=True))


def test_remesh_rebuilds_h5m_and_preserves_material(tmp_path, sphere_path):
    pytest.importorskip('gpytoolbox')
    from surface_mesh.mesh import load_mesh
    from surface_mesh.int_optimizer.remesh import remesh_h5m
    from surface_mesh.mesh.moab_adapter import MoabMesh
    from surface_mesh.mesh.reporting import file_hash
    tagged = tmp_path / 'tagged.h5m'
    original = MoabMesh(sphere_path.with_suffix('.h5m'))
    original.dagmc_model().volumes[0].material = 'iron'
    original.write(tagged)
    source = load_mesh(tagged)
    output = tmp_path / 'remeshed.h5m'
    info = remesh_h5m(source, output, IntConfig(remesh_target_edge_length=.3))
    actual = load_mesh(output)
    assert info['vertices_after'] != info['vertices_before']
    assert len(actual.coordinates) == info['vertices_after']
    assert (actual.surface_id, actual.volume_id) == (source.surface_id, source.volume_id)
    assert MoabMesh(output).dagmc_model().volumes[0].material == 'iron'
    assert file_hash(tagged) == source.input_sha256
    assert info['project'] is True
    with pytest.raises(FileExistsError):
        remesh_h5m(source, output, IntConfig())


def test_volume_target_survives_mesh_change(mesh):
    x, f = mesh
    target = signed_volume(x, f)
    smaller = .99 * x
    result = optimize_vertices(smaller, f, np.zeros_like(x), IntConfig(iterations=10),
                               volume_target=target)
    assert result.status == 'completed'
    assert result.diagnostics['initial_volume'] == target
    assert abs(signed_volume(result.vertices, f) - target) < abs(signed_volume(smaller, f) - target)


def test_persisted_adam_state_damps_reversed_gradient(mesh):
    x, f = mesh
    g = np.zeros_like(x)
    g[:, 1] = 1e-3
    config = IntConfig(iterations=1, learning_rate=.01, volume_weight=0.)
    first = optimize_vertices(x, f, g, config)
    assert first.adam_state['t'] == 1
    assert first.diagnostics['adam_state_transfer'] is None
    fresh = optimize_vertices(first.vertices, f, -g, config)
    kept = optimize_vertices(first.vertices, f, -g, config, adam_state=first.adam_state)
    assert kept.adam_state['t'] == 2
    assert kept.diagnostics['adam_state_transfer']['max_transfer_distance'] == 0.
    fresh_dy = np.mean(fresh.vertices[:, 1] - first.vertices[:, 1])
    kept_dy = np.mean(kept.vertices[:, 1] - first.vertices[:, 1])
    assert fresh_dy > 0 and 0 <= kept_dy < .1 * fresh_dy


def test_adam_state_transfers_to_remeshed_vertices(mesh):
    from surface_mesh.int_optimizer.engine import transfer_adam_state
    x, f = mesh
    state = dict(vertices=x, m=x.copy(), v=np.abs(x), t=7)
    finer = sphere(2)[0]
    m, v, t, distance = transfer_adam_state(state, finer)
    assert m.shape == v.shape == finer.shape and t == 7 and distance > 0
    # m copies source positions, so each row must be a nearest source vertex.
    nearest = np.linalg.norm(finer[:, None] - x[None], axis=2).min(axis=1)
    np.testing.assert_allclose(np.linalg.norm(finer - m, axis=1), nearest)
    with pytest.raises(ValueError):
        transfer_adam_state(dict(state, v=-np.abs(x)), finer)
    with pytest.raises(ValueError):
        IntConfig(persist_adam_state=1)


def test_int_repo_remesh_options_validate():
    config = IntConfig(remesh_adam_state='reset', remesh_learning_rate_per_edge=.07,
                       remesh_halve_edge_rounds=[3, 5])
    assert config.remesh_halve_edge_rounds == (3, 5)
    for bad in (dict(remesh_adam_state='keep'), dict(remesh_learning_rate_per_edge=0.),
                dict(remesh_halve_edge_rounds=(0,)), dict(remesh_halve_edge_rounds=(True,))):
        with pytest.raises(ValueError):
            IntConfig(**bad)


def test_chained_persisted_steps_match_one_solve():
    # A non-uniform gradient exposes any mix-up between LargeSteps' sorted rows
    # and input vertex order; steps move vertices past their sort neighbors.
    x, f = sphere(2, 2.0)
    g = np.random.default_rng(1).normal(size=x.shape) * 1e-3
    config = IntConfig(large_steps_lambda=35., learning_rate=.05, iterations=1,
                       volume_weight=5e-6, range_weight=0.)
    first = optimize_vertices(x, f, g, config)
    second = optimize_vertices(first.vertices, f, g, config, adam_state=first.adam_state)
    reference = optimize_vertices(x, f, g, replace(config, iterations=2))
    np.testing.assert_allclose(second.vertices, reference.vertices, atol=1e-5)
    np.testing.assert_allclose(second.adam_state['m'], reference.adam_state['m'], rtol=1e-4, atol=1e-9)
