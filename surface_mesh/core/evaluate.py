"""Evaluate each mesh in a fresh OpenMC process and retain its derivative arrays."""

from dataclasses import asdict
import json
import os
from pathlib import Path
import sys

import numpy as np

from .runtime import atomic_write, configure_trigger
from .progress import phase, run_logged


def validate_surface_derivative(objective, sample):
    """Do not mistake an unpopulated surface derivative for stationarity."""
    import openmc
    if (any(isinstance(f, openmc.SurfaceFilter) for f in objective.filters)
            and all(not np.any(sample[key])
                    for key in ('geometry_derivative', 'dJ', 'datt'))):
        raise ValueError(
            'Surface objective returned all-zero geometric derivatives. Check that '
            'OpenMC score_surface_tally calls tally_geometry_derivatives, and that '
            'enough histories sample the differentiated mesh; this is not evidence '
            'of optimization convergence.')


def verify_vertex_order(snapshot, vertices):
    """Require exact, unambiguous correspondence to XDG's exported vertex order."""
    if (vertices.shape != snapshot.coordinates.shape
            or not np.array_equal(vertices, snapshot.coordinates)
            or len(np.unique(vertices, axis=0)) != len(vertices)):
        raise ValueError('XDG/PyMOAB vertex order differs or coordinates are ambiguous')


def verify_sample_mesh(snapshot, sample):
    """Refuse a gradient computed on any other geometry or file revision."""
    if str(sample['input_sha256']) != snapshot.input_sha256:
        raise ValueError('Derivative sample belongs to a different mesh file')
    verify_vertex_order(snapshot, sample['coordinates'])


def validate_model_container(model, settings):
    """Check that the model's actual CSG regions match deformation bounds."""
    if settings.model_variant == 'box':
        cells = model.geometry.root_universe.cells
        container, water = cells[20].region.bounding_box, cells[21].region.bounding_box
        translation = np.zeros(3) if cells[20].translation is None else cells[20].translation
        for actual, expected in zip(container, (settings.box_container_lower, settings.box_container_upper)):
            if not np.allclose(np.asarray(actual) - translation, expected, rtol=0, atol=1e-12):
                raise ValueError('Model and deformation box bounds differ')
        if np.all(np.minimum(container[1], water[1]) > np.maximum(container[0], water[0]) + 1e-12):
            raise ValueError('The mesh container overlaps the water box')
        air = cells[23].region.bounding_box
        air_center = (air[0] + air[1]) / 2
        if air_center in cells[20].region:
            raise ValueError('The mesh container overlaps the source air sphere')
        return
    surfaces = model.geometry.get_all_surfaces()
    container, water = surfaces[20], surfaces[21]
    center = np.array([container.x0, container.y0, container.z0])
    water_center = np.array([water.x0, water.y0, water.z0])
    cell = model.geometry.root_universe.cells[20]
    translation = np.zeros(3) if cell.translation is None else cell.translation
    if not np.allclose(center, translation, rtol=0, atol=1e-12):
        raise ValueError('Mesh translation must match its spherical container center')
    if not np.isclose(container.r, settings.container_radius, rtol=0, atol=1e-12):
        raise ValueError('Model and deformation container radii differ')
    if container.r + water.r > np.linalg.norm(center - water_center) + 1e-12:
        raise ValueError('The translated CSG container overlaps the water sphere')


def evaluate(mesh_path, directory, settings, seed, budget=1, *, batch_scores=False,
             settings_kind='int'):
    """Run transport in isolation; crashes leave a log and cannot corrupt the driver."""
    if settings_kind != 'int':
        raise ValueError('Unknown evaluation settings kind')
    directory = Path(directory).resolve()
    batch_scores = batch_scores or settings.paired_sampling
    directory.mkdir(parents=True, exist_ok=False)
    mesh_path = Path(mesh_path).resolve()
    settings_path = directory / 'settings.json'
    settings_path.write_text(json.dumps(asdict(settings), indent=2) + '\n')
    env = os.environ.copy()
    env['OMP_NUM_THREADS'] = str(settings.threads)
    if not env.get('OPENMC_CROSS_SECTIONS'):
        fallback = Path.home() / 'nndc_hdf5/cross_sections.xml'
        if not fallback.is_file():
            raise RuntimeError('Set OPENMC_CROSS_SECTIONS to the NNDC cross_sections.xml')
        env['OPENMC_CROSS_SECTIONS'] = str(fallback)
    root = Path(__file__).resolve().parents[2]
    with (directory / 'transport.log').open('w') as log:
        command = (
            [sys.executable, '-u', '-m', 'surface_mesh.cli.evaluate', '--mesh', str(mesh_path),
             '--directory', str(directory), '--settings', str(settings_path),
             '--seed', str(seed), '--budget', str(budget)]
            + (['--batch-scores'] if batch_scores else [])
            + (['--settings-kind', 'int'] if settings_kind == 'int' else []))
        returncode = run_logged(command, log, cwd=root, env=env, label=directory.name)
    if returncode:
        raise RuntimeError(f'OpenMC evaluation failed ({returncode}); see {directory / "transport.log"}')
    with np.load(directory / 'sample.npz', allow_pickle=False) as sample:
        return {name: sample[name].copy() for name in sample.files}


def worker(mesh_path, directory, settings, seed, budget, *, batch_scores=False, progress=False):
    import h5py
    import openmc
    import openmc.lib
    from surface_mesh.mesh import load_mesh
    from .models import build_model, objective_tally

    snapshot = load_mesh(mesh_path, validate_geometry=getattr(settings, 'validate_geometry', True))
    if snapshot.surface_id != 1 or snapshot.volume_id != 1:
        raise ValueError('Initial workflow requires surface/volume IDs 1 to match the model')
    if not openmc.lib.feature_enabled('xdg'):
        raise RuntimeError('OpenMC must be built with XDG support')
    openmc.reset_auto_ids()
    model = build_model(settings, snapshot.path, seed=seed, budget=budget)
    objective = objective_tally(model)
    if getattr(settings, 'validate_geometry', True):
        validate_model_container(model, settings)
    configure_trigger(model, 1, settings.flux_rel_err,
                      settings.trigger_max_factor, settings.trigger_interval)
    # Unique process + directory provides fresh mesh acceleration and ID state.
    os.chdir(directory)
    model.export_to_model_xml()
    if progress:
        print('[INT progress] Initializing OpenMC ...', flush=True)
    with openmc.lib.run_in_memory(args=['-s', str(settings.threads)]):
        if settings.save_geometry_images:
            from .plots import save_geometry_image
            with phase('Rendering material slices', enabled=progress):
                save_geometry_image(model, snapshot.path, settings, 'geometry.png')
        if settings.save_solid_geometry_images:
            from .plots import save_solid_geometry_image
            with phase('Rendering solid geometry', enabled=progress):
                save_solid_geometry_image(model, settings, 'geometry_solid.png')
        if progress:
            print(f'[INT progress] Transport: {settings.particles * budget} particles/batch, '
                  f'{settings.batches} requested batches, '
                  f'{settings.batches * settings.trigger_max_factor} maximum batches', flush=True)
        if batch_scores:
            # Fixed-source, fixed-budget batches permit paired comparisons of
            # nearby meshes with the same source-history random seeds.
            if model.settings.run_mode != 'fixed source' or settings.trigger_max_factor != 1:
                raise ValueError('Batch scoring requires fixed-source, fixed-budget transport')
            flux_batches = []
            previous_sum = 0.
            openmc.lib.simulation_init()
            for _ in openmc.lib.iter_batches():
                tally = openmc.lib.tallies[1]
                if tally.num_realizations != len(flux_batches) + 1 or tally.mean.size != 1:
                    raise ValueError('Expected one scalar realization per batch')
                current_sum = float(tally.results[:, :, 1].flat[0])
                flux_batches.append(current_sum - previous_sum)
                previous_sum = current_sum
            openmc.lib.simulation_finalize()
        else:
            openmc.lib.run()
        if progress:
            print('[INT progress] Saving tally, derivatives, and statepoint ...', flush=True)
        tally = openmc.lib.tallies[1]
        derivative = openmc.lib.geometry_derivatives[1]
        n = tally.num_realizations
        if n < 2 or tally.mean.size != 1:
            raise ValueError('Need >=2 realizations and one objective tally bin')
        if derivative.mean.shape != (snapshot.coordinates.size, 1):
            raise ValueError('Expected exactly three derivative rows per mesh vertex')
        sample = {
            'tally_mean': float(tally.mean.flat[0]),
            'tally_std_dev': float(tally.std_dev.flat[0]),
            'geometry_derivative': derivative.mean[:, 0].copy().reshape(-1, 3),
            'dJ': (derivative.dJ[:, 0] / n).copy().reshape(-1, 3),
            'datt': (derivative.datt[:, 0] / n).copy().reshape(-1, 3),
            'coordinates': snapshot.coordinates, 'vertex_handles': snapshot.vertex_handles,
            'faces': snapshot.faces, 'num_realizations': n, 'seed': seed,
            'particles': settings.particles * budget, 'input_sha256': snapshot.input_sha256,
            'openmc_version': openmc.__version__,
            'cross_sections': str(openmc.config['cross_sections']),
        }
        if batch_scores:
            sample['batch_flux'] = np.asarray(flux_batches)
            np.testing.assert_allclose(sample['batch_flux'].mean(), sample['tally_mean'],
                                       rtol=1e-10, atol=1e-14)
            np.testing.assert_allclose(sample['batch_flux'].std(ddof=1) / np.sqrt(n),
                                       sample['tally_std_dev'], rtol=1e-7, atol=1e-14)
        openmc.lib.statepoint_write('evaluation.h5', write_source=False)
        with h5py.File('evaluation.h5') as statepoint:
            vertices = statepoint['tallies/meshes/mesh 1/vertices'][()]
        verify_vertex_order(snapshot, vertices)
        for key in ('tally_mean', 'tally_std_dev', 'geometry_derivative', 'dJ', 'datt'):
            if not np.isfinite(sample[key]).all():
                raise ValueError(f'Nonfinite transport result: {key}')
        if objective.scores == ['flux'] and sample['tally_mean'] <= 0:
            raise ValueError('No positive flux estimate; increase the sampling budget')
        validate_surface_derivative(objective, sample)
        if settings.save_geometry_images:
            sample['geometry_image'] = 'geometry.png'
        if settings.save_solid_geometry_images:
            sample['geometry_solid_image'] = 'geometry_solid.png'
        atomic_write('sample.npz', lambda stream: np.savez_compressed(stream, **sample))
