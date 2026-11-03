"""Vertex diagnostics in legacy VTK format, using the mesh's local coordinates."""

import io
import json
from pathlib import Path

import numpy as np
from surface_mesh.mesh import load_mesh

from .evaluate import verify_sample_mesh
from .runtime import atomic_write


def write_vtk(path, coordinates, faces, vectors, scalars=None):
    """Write triangle geometry and full-precision point arrays without VTK dependencies."""
    coordinates, faces = np.asarray(coordinates), np.asarray(faces)
    n = len(coordinates)
    if coordinates.shape != (n, 3) or not np.isfinite(coordinates).all():
        raise ValueError('Expected finite XYZ vertices')
    if (faces.ndim != 2 or faces.shape[1] != 3 or faces.dtype.kind not in 'iu'
            or np.any(faces < 0) or np.any(faces >= n)):
        raise ValueError('Expected valid triangle indices')
    scalars = {'vertex_index': np.arange(n), **(scalars or {})}
    for fields, shape in ((vectors, (n, 3)), (scalars, (n,))):
        for name, values in fields.items():
            if (not name or any(c.isspace() for c in name)
                    or np.shape(values) != shape or not np.isfinite(values).all()):
                raise ValueError(f'Invalid VTK point field: {name}')
    stream = io.StringIO()
    stream.write('# vtk DataFile Version 3.0\nOpenMC vertex diagnostics; local mesh frame\n'
                 'ASCII\nDATASET POLYDATA\n')
    stream.write(f'POINTS {n} double\n')
    np.savetxt(stream, coordinates, fmt='%.17g')
    stream.write(f'POLYGONS {len(faces)} {4 * len(faces)}\n')
    np.savetxt(stream, np.column_stack((np.full(len(faces), 3), faces)), fmt='%d')
    stream.write(f'POINT_DATA {n}\n')
    for name, values in vectors.items():
        stream.write(f'VECTORS {name} double\n')
        np.savetxt(stream, values, fmt='%.17g')
    for name, values in scalars.items():
        integer = np.asarray(values).dtype.kind in 'iu'
        stream.write(f'SCALARS {name} {"int" if integer else "double"} 1\nLOOKUP_TABLE default\n')
        np.savetxt(stream, values, fmt='%d' if integer else '%.17g')
    encoded = stream.getvalue().encode()
    atomic_write(path, lambda output: output.write(encoded))


def vector_magnitudes(vectors):
    return {f'{name}_magnitude': np.linalg.norm(value, axis=1)
            for name, value in vectors.items()}


def export_evaluation(mesh_path, sample, output, *, validate_geometry=True):
    """Export the raw derivative on the exact mesh used in this evaluation."""
    mesh = load_mesh(mesh_path, validate_geometry=validate_geometry)
    verify_sample_mesh(mesh, sample)
    gradient = np.asarray(sample['geometry_derivative'])
    vectors = {'derivative': gradient, 'descent_derivative': -gradient}
    for component in ('dJ', 'datt'):
        if component in sample:
            vectors[component] = sample[component]
    write_vtk(output, mesh.coordinates, mesh.faces, vectors, vector_magnitudes(vectors))


def export_trial(parent_path, candidate_path, arrays, output, *, validate_geometry=True):
    """Attach the generating gradient and realized motion to the parent vertices."""
    parent = load_mesh(parent_path, validate_geometry=validate_geometry)
    candidate = load_mesh(candidate_path, validate_geometry=validate_geometry)
    if (parent.coordinates.shape != candidate.coordinates.shape
            or not np.array_equal(parent.faces, candidate.faces)):
        raise ValueError('Trial VTK requires identical parent/candidate topology')
    applied = np.asarray(arrays['applied_displacement'])
    if (applied.shape != parent.coordinates.shape or not np.allclose(
            parent.coordinates + applied, candidate.coordinates, rtol=0, atol=1e-12)):
        raise ValueError('Saved displacement does not reproduce the trial mesh')
    gradient = np.asarray(arrays['gradient'])
    vectors = {'derivative': gradient, 'descent_derivative': -gradient,
               'search_direction': arrays['direction'], 'displacement': applied}
    scalars = vector_magnitudes(vectors)
    contribution = np.sum(gradient * applied, axis=1)
    denominator = scalars['derivative_magnitude'] * scalars['displacement_magnitude']
    defined = denominator > 0
    alignment = np.zeros(len(applied))
    np.divide(-contribution, denominator, out=alignment, where=defined)
    scalars.update(predicted_objective_change=contribution,
                   descent_alignment=np.clip(alignment, -1., 1.),
                   descent_alignment_defined=defined.astype(int))
    write_vtk(output, parent.coordinates, parent.faces, vectors, scalars)


def export_series(root, history):
    """Index accepted shapes with derivatives evaluated on those exact shapes.

    Use the latest completed evaluation of each mesh (including resampling).
    Missing evaluations are omitted rather than borrowing another shape's gradient.
    Times are accepted-step indices, starting with the initial mesh at zero.
    """
    root = Path(root)
    evaluations = {entry['mesh']: entry for entry in history['evaluations']
                   if entry['status'] == 'complete' and entry.get('vtk')}
    files = []
    for step, mesh in enumerate(history['accepted_sequence']):
        entry = evaluations.get(mesh)
        if entry is not None and (root / entry['vtk']).is_file():
            files.append({'name': entry['vtk'], 'time': step})
    if files:
        encoded = (json.dumps({'file-series-version': '1.0', 'files': files},
                              indent=2) + '\n').encode()
        atomic_write(root / 'mesh_evolution.vtk.series', lambda stream: stream.write(encoded))


def export_run(root):
    """Backfill completed evaluations and feasible trials without rerunning transport."""
    root = Path(root).resolve()
    history_path = root / 'history.json'
    history = json.loads(history_path.read_text())
    if history['status'] == 'running':
        raise ValueError('Backfill requires a stopped run to avoid concurrent history edits')
    count = 0
    validate_geometry = history.get('int_inspired', {}).get('enable_safeguards',
                                                         history.get('mode') != 'int_inspired')
    for entry in history['evaluations']:
        if entry['status'] != 'complete':
            continue
        directory = root / entry['directory']
        output = directory / 'derivatives.vtk'
        with np.load(directory / 'sample.npz', allow_pickle=False) as sample:
            export_evaluation(root / entry['mesh'], sample, output, validate_geometry=validate_geometry)
        entry['vtk'] = str(output.relative_to(root))
        count += 1
    records = history.get('trials', []) + history.get('sweep', [])
    for entry in records:
        if not entry.get('mesh') or entry['mesh'] == history['initial_mesh']:
            continue
        candidate = root / entry.get('optimization_mesh', entry['mesh'])
        array_path = candidate.with_suffix('.npz')
        if not array_path.is_file():
            continue
        output = candidate.with_name(candidate.stem + '_vectors.vtk')
        parent = root / entry.get('parent_mesh', history['initial_mesh'])
        with np.load(array_path, allow_pickle=False) as arrays:
            fields = dict(arrays)
            if 'amplitude' in entry:
                fields['direction'] = entry['amplitude'] * fields['direction']
            export_trial(parent, candidate, fields, output, validate_geometry=validate_geometry)
        entry['vtk'] = str(output.relative_to(root))
        count += 1
    export_series(root, history)
    encoded = (json.dumps(history, indent=2, allow_nan=False) + '\n').encode()
    atomic_write(history_path, lambda output: output.write(encoded))
    return count
