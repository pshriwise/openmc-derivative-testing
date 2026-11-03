"""Independent INT launch, derivative refresh, and shared visual artifacts."""

from dataclasses import replace
import json
from pathlib import Path
import sys

import numpy as np
import pytest
from PIL import Image

pytest.importorskip('mitsuba')
pytest.importorskip('cholespy')
pytest.importorskip('pymoab')

from surface_mesh.mesh import load_mesh
from surface_mesh.int_optimizer import IntConfig
from ..core.settings import IntRunSettings, load_run_config
from ..core.run import run
from ..core.report import regenerate
from ..core.vtk_output import export_run

INPUT = Path(__file__).resolve().parents[1].joinpath('sphere_coarse.h5m')


def evaluator(mesh_path, directory, cfg, seed, budget):
    directory.mkdir()
    mesh = load_mesh(mesh_path)
    x = mesh.coordinates
    gradient = np.zeros_like(x)
    gradient[:, 0] = 1. + .001 * x[:, 0]
    sample = dict(tally_mean=float(np.sum(x[:, 0] + .0005 * x[:, 0]**2)),
                  tally_std_dev=.001, geometry_derivative=gradient,
                  coordinates=x, faces=mesh.faces, vertex_handles=mesh.vertex_handles,
                  input_sha256=mesh.input_sha256, particles=cfg.particles,
                  num_realizations=cfg.batches)
    for flag, field, filename in [('save_geometry_images', 'geometry_image', 'geometry.png'),
                                  ('save_solid_geometry_images', 'geometry_solid_image', 'geometry_solid.png')]:
        if getattr(cfg, flag):
            Image.new('RGB', (120, 100), (seed % 255, 90, 150)).save(directory / filename)
            sample[field] = filename
    np.savez(directory / 'sample.npz', **sample)
    return sample


def test_fresh_int_run_uses_shared_visualizations(tmp_path):
    root = tmp_path / 'int'
    # The translated sphere model needs a container that clears the water sphere.
    cfg = IntRunSettings(max_iterations=2, source_clearance=0., container_radius=10.)
    result = run(INPUT, root, IntConfig(iterations=1), cfg, evaluator=evaluator)
    assert result['mode'] == 'int_inspired'
    assert result['status'] == 'iteration_limit'
    assert len(result['accepted_sequence']) == 3
    assert len(result['evaluations']) == 3
    assert 'smoothing' not in result['settings']
    assert result['int_inspired']['iterations'] == 1
    assert result['int_run']['max_iterations'] == 2
    assert result['trials'][1]['gradient_evaluation'] == 1
    assert result['trials'][1]['parent_mesh'] == result['accepted_sequence'][1]
    for trial in result['trials']:
        parent = load_mesh(root / trial['parent_mesh'])
        candidate = load_mesh(root / trial['mesh'])
        with np.load((root / trial['mesh']).with_suffix('.npz')) as arrays:
            np.testing.assert_allclose(parent.coordinates + arrays['applied_displacement'],
                                       candidate.coordinates, atol=1e-14, rtol=0)
            np.testing.assert_allclose(arrays['gradient'][:, 0], 1 + .001 * parent.coordinates[:, 0])
        assert (root / trial['vtk']).exists()
    for entry in result['evaluations']:
        for key in ('image', 'solid_image', 'vtk'):
            assert (root / entry[key]).is_file()
    for name in ('accepted_geometry.gif', 'accepted_geometry_solid.gif'):
        with Image.open(root / name) as gif:
            assert gif.n_frames == 3
    for extension in ('png', 'svg', 'csv'):
        assert (root / f'objective_history.{extension}').is_file()
    assert (root / 'mesh_motion.png').is_file()
    motion = json.loads((root / 'mesh_motion.json').read_text())
    assert motion['meshes'] == result['accepted_sequence']
    assert motion['motion'][0]['max_motion_mm'] == 0.
    initial = load_mesh(root / result['accepted_sequence'][0])
    final = load_mesh(root / result['accepted_sequence'][-1])
    assert motion['motion'][-1]['max_motion_mm'] == pytest.approx(
        10 * np.linalg.norm(final.coordinates - initial.coordinates, axis=1).max())
    frames = json.loads((root / 'animation_frames.json').read_text())
    assert [frame['mesh'] for frame in frames] == result['accepted_sequence']
    before = {p: p.read_bytes() for p in root.rglob('*.vtk')}
    assert export_run(root) == 5
    assert all(p.read_bytes() == content for p, content in before.items())
    regenerate(root)
    with pytest.raises(FileExistsError):
        run(INPUT, root, IntConfig(), cfg, evaluator=evaluator)


def test_int_geometry_rejection_keeps_baseline(tmp_path):
    root = tmp_path / 'reject'
    result = run(INPUT, root, IntConfig(iterations=1, max_displacement=1e-8, enable_safeguards=True),
                 IntRunSettings(source_clearance=0., container_radius=10.), evaluator=evaluator)
    assert result['status'] == 'geometry_rejected'
    assert len(result['evaluations']) == 1
    assert result['accepted_mesh'] == 'initial.h5m'
    assert not list(root.glob('iteration*.h5m'))
    assert (root / 'accepted_geometry.gif').exists()


def test_failed_candidate_transport_is_not_accepted(tmp_path):
    def failing(mesh_path, directory, cfg, seed, budget):
        if mesh_path.name != 'initial.h5m':
            raise RuntimeError('Injected transport failure')
        return evaluator(mesh_path, directory, cfg, seed, budget)
    root = tmp_path / 'failed'
    with pytest.raises(RuntimeError, match='Injected'):
        run(INPUT, root, IntConfig(iterations=1),
            IntRunSettings(source_clearance=0., container_radius=10.), evaluator=failing)
    history = json.loads((root / 'history.json').read_text())
    assert history['status'] == 'failed'
    assert history['accepted_sequence'] == ['initial.h5m']
    assert history['evaluations'][-1]['status'] == 'failed'


def test_int_run_config_rejects_unrelated_settings(tmp_path):
    assert IntRunSettings().particles == 10000
    path = tmp_path / 'settings.json'
    path.write_text('{"int_run": {"particles": 2000, "max_iterations": 3}}')
    assert load_run_config(path).particles == 2000
    path.write_text('{"int_run": {"smoothing": 1}}')
    with pytest.raises(TypeError):
        load_run_config(path)
    path.write_text('{"int_inspired": {"iterations": 3}}')
    with pytest.raises(ValueError, match='int_run'):
        load_run_config(path)
    with pytest.raises(TypeError, match='IntRunSettings'):
        run(INPUT, tmp_path / 'wrong', IntConfig(), object(), evaluator=evaluator)


def test_worker_selects_int_settings(tmp_path, monkeypatch):
    from dataclasses import asdict
    from ..cli import evaluate
    path = tmp_path / 'settings.json'
    path.write_text(json.dumps(asdict(IntRunSettings())))
    seen = []
    monkeypatch.setattr(evaluate.evaluate, 'worker', lambda mesh, directory, cfg, *a, **k: seen.append(cfg))
    monkeypatch.setattr(sys, 'argv', ['evaluate', '--mesh', str(INPUT), '--directory', str(tmp_path),
                                    '--settings', str(path), '--seed', '1', '--settings-kind', 'int'])
    evaluate.main()
    assert isinstance(seen[0], IntRunSettings)


def test_remeshed_run_refreshes_gradients_and_visualizations(tmp_path):
    pytest.importorskip('gpytoolbox')
    root = tmp_path / 'remeshed'
    cfg = IntRunSettings(max_iterations=2, model_variant='box', source_clearance=100.)
    optimizer = IntConfig(iterations=1, remesh_interval=1, max_displacement=1e-12,
                          remesh_target_edge_length=.4)
    result = run(INPUT, root, optimizer, cfg, evaluator=evaluator)
    assert result['status'] == 'iteration_limit'
    assert len(result['evaluations']) == 3
    assert not result['int_run']['validate_geometry']
    for i, trial in enumerate(result['trials']):
        assert trial['gradient_evaluation'] == i
        parent = load_mesh(root / trial['parent_mesh'])
        candidate = load_mesh(root / trial['mesh'])
        assert trial['remeshing']['vertices_after'] == len(candidate.coordinates)
        with np.load((root / trial['optimization_mesh']).with_suffix('.npz')) as arrays:
            assert arrays['gradient'].shape == parent.coordinates.shape
            np.testing.assert_allclose(arrays['gradient'][:, 0], 1 + .001 * parent.coordinates[:, 0])
        with np.load(root / result['evaluations'][i+1]['directory'] / 'sample.npz') as sample:
            np.testing.assert_array_equal(sample['coordinates'], candidate.coordinates)
    assert result['trials'][0]['remeshing']['vertices_before'] != result['trials'][0]['remeshing']['vertices_after']
    motion = json.loads((root / 'mesh_motion.json').read_text())['motion'][-1]
    assert motion['max_motion_mm'] is None
    assert motion['max_sampled_surface_distance_mm'] > 0
    before = {p: p.read_bytes() for p in root.rglob('*.vtk')}
    assert export_run(root) == 5
    assert all(p.read_bytes() == content for p, content in before.items())
    regenerate(root)
    for name in ('accepted_geometry.gif', 'accepted_geometry_solid.gif'):
        with Image.open(root / name) as gif:
            assert gif.n_frames == 3


def test_adaptive_sampling_refreshes_parent_before_optimization(tmp_path):
    from ..core.sampling import AdaptiveSamplingConfig
    from ..core.report import accepted_evaluations
    calls = []

    def plateau(mesh, directory, cfg, seed, budget):
        calls.append((mesh.name, cfg.particles))
        result = evaluator(mesh, directory, cfg, seed, budget)
        result['tally_mean'] = 1.
        result['geometry_derivative'] *= cfg.particles / 100
        np.savez(directory / 'sample.npz', **result)
        return result

    cfg = IntRunSettings(max_iterations=2, model_variant='box', particles=100,
                        adaptive_sampling=AdaptiveSamplingConfig(enabled=True, start_iteration=1,
                            patience=1, window=2, max_particles=200))
    root = tmp_path / 'adaptive'
    h = run(INPUT, root, IntConfig(iterations=1, remesh_interval=1), cfg, evaluator=plateau)
    assert [n for _, n in calls] == [100, 100, 200, 200]
    assert calls[1][0] == calls[2][0]  # Same accepted geometry, fresh precision.
    assert [e['particles'] for e in h['evaluations']] == [100, 100, 200, 200]
    assert h['trials'][1]['gradient_evaluation'] == 2
    assert [(step, e['index']) for step, e in accepted_evaluations(h)] == [(0, 0), (1, 2), (2, 3)]
    trial = h['trials'][1]
    with np.load((root / trial['optimization_mesh']).with_suffix('.npz')) as a:
        with np.load(root / 'evaluation_0002/sample.npz') as fresh:
            np.testing.assert_array_equal(a['gradient'], fresh['geometry_derivative'])
    assert h['adaptive_sampling']['decisions'][1]['status'] == 'refreshed'
    assert h['evaluations'][2]['purpose'] == 'adaptive_sampling_refresh'
    assert h['int_run']['particles'] == 100  # Immutable initial configuration.
    assert json.loads((root / 'completion_summary.json').read_text())['final_particles'] == 200
    assert export_run(root) == 6
    regenerate(root)
    with Image.open(root / 'accepted_geometry.gif') as gif:
        assert gif.n_frames == 3


def test_adaptive_refresh_failure_does_not_take_another_shape_step(tmp_path):
    from ..core.sampling import AdaptiveSamplingConfig

    def fail_refresh(mesh, directory, cfg, seed, budget):
        if cfg.particles > 100:
            raise RuntimeError('Injected refresh failure')
        s = evaluator(mesh, directory, cfg, seed, budget)
        s['tally_mean'] = 1.
        np.savez(directory / 'sample.npz', **s)
        return s

    cfg = IntRunSettings(max_iterations=2, model_variant='box', particles=100,
                        adaptive_sampling=AdaptiveSamplingConfig(enabled=True, start_iteration=1,
                            patience=1, window=2, max_particles=200))
    root = tmp_path / 'failed_refresh'
    with pytest.raises(RuntimeError, match='Injected refresh'):
        run(INPUT, root, IntConfig(iterations=1), cfg, evaluator=fail_refresh)
    h = json.loads((root / 'history.json').read_text())
    assert h['status'] == 'failed'
    assert len(h['accepted_sequence']) == 2
    assert len(h['trials']) == 1
    assert h['evaluations'][-1]['status'] == 'failed'


def test_persisted_adam_state_crosses_remeshed_updates(tmp_path):
    pytest.importorskip('gpytoolbox')
    root = tmp_path / 'persisted'
    cfg = IntRunSettings(max_iterations=3, model_variant='box', source_clearance=100.)
    optimizer = IntConfig(iterations=2, remesh_interval=1, remesh_target_edge_length=.4,
                          persist_adam_state=True)
    result = run(INPUT, root, optimizer, cfg, evaluator=evaluator)
    assert result['status'] == 'iteration_limit'
    transfers = [json.loads((root / t['int_history']).read_text())['diagnostics']['adam_state_transfer']
                 for t in result['trials']]
    assert transfers[0] is None
    assert [t['previous_steps'] for t in transfers[1:]] == [2, 4]
    assert all(t['max_transfer_distance'] >= 0 for t in transfers[1:])


def test_int_repo_remesh_resets_adam_and_rescales_learning_rate(tmp_path):
    pytest.importorskip('gpytoolbox')
    root = tmp_path / 'int_repo'
    cfg = IntRunSettings(max_iterations=3, model_variant='box', source_clearance=100.)
    optimizer = IntConfig(iterations=1, remesh_interval=1, persist_adam_state=True,
                          remesh_adam_state='reset', remesh_learning_rate_per_edge=.07,
                          remesh_halve_edge_rounds=(2,))
    result = run(INPUT, root, optimizer, cfg, evaluator=evaluator)
    trials = result['trials']
    transfers = [json.loads((root / t['int_history']).read_text())['diagnostics']['adam_state_transfer']
                 for t in trials]
    assert transfers == [None, None, None]
    assert trials[0]['learning_rate'] == optimizer.learning_rate
    for before, after in zip(trials, trials[1:]):
        rate = .07 * before['remeshing']['target_edge_length']
        assert after['learning_rate'] == pytest.approx(rate)
        assert before['remeshing']['next_learning_rate'] == pytest.approx(rate)
    halved = load_mesh(root / trials[1]['optimization_mesh'])
    f, x = halved.faces, halved.coordinates
    edges = np.concatenate((f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]))
    mean = np.linalg.norm(x[edges[:, 0]] - x[edges[:, 1]], axis=1).mean()
    assert trials[1]['remeshing']['target_edge_length'] == pytest.approx(.5 * mean)


@pytest.mark.parametrize('saved', [True, False])
def test_continuation_matches_uninterrupted_run(tmp_path, saved):
    from ..core.continuation import REPLAY_TOLERANCE_CM, continuation_state
    optimizer = IntConfig(iterations=1, learning_rate=.01, persist_adam_state=True)
    cfg = IntRunSettings(max_iterations=5, model_variant='box', source_clearance=100.)
    whole = run(INPUT, tmp_path / 'whole', optimizer, cfg, evaluator=evaluator)
    first = run(INPUT, tmp_path / 'first', optimizer, replace(cfg, max_iterations=3), evaluator=evaluator)
    if not saved:  # A run from before Adam states were saved: rebuild by replay.
        manifest = tmp_path / 'first' / 'history.json'
        data = json.loads(manifest.read_text())
        for trial in data['trials']:
            (tmp_path / 'first' / trial.pop('adam_state')).unlink()
        manifest.write_text(json.dumps(data))
    state = continuation_state(tmp_path / 'first', optimizer, cfg)
    assert state.provenance['adam_state'] == ('saved' if saved else 'replayed')
    assert state.next_iteration == 4 and state.adam_state['t'] == 3
    assert state.seed == cfg.seed + 104729 * len(first['evaluations'])
    resumed = run(state.mesh, tmp_path / 'resumed', optimizer, cfg, evaluator=evaluator, continuation=state)
    assert [t['iteration'] for t in resumed['trials']] == [4, 5]
    assert resumed['continued_from']['last_accepted_iteration'] == 3
    # The LLVM backend differs by about half a float32 ulp between identical calls.
    np.testing.assert_allclose(load_mesh(tmp_path / 'resumed' / resumed['accepted_mesh']).coordinates,
                               load_mesh(tmp_path / 'whole' / whole['accepted_mesh']).coordinates,
                               atol=REPLAY_TOLERANCE_CM)
    with pytest.raises(ValueError):
        run(state.mesh, tmp_path / 'short', optimizer, replace(cfg, max_iterations=3),
            evaluator=evaluator, continuation=state)
