"""Show actual mesh motion at a close scale, without magnifying displacement."""

import json
from pathlib import Path

import numpy as np


def section_segments(vertices, faces, normal, level):
    """Intersect triangles with a fixed coordinate plane."""
    axes = [axis for axis in range(3) if axis != normal]
    segments = []
    for triangle in vertices[faces]:
        points = []
        for i, j in ((0, 1), (1, 2), (2, 0)):
            a, b = triangle[i], triangle[j]
            da, db = a[normal] - level, b[normal] - level
            if da == db:
                if da == 0:
                    points.extend((a, b))
            elif da * db <= 0:
                points.append(a + (b - a) * da / (da - db))
        if len(points) >= 2:
            points = np.unique(points, axis=0)
            if len(points) >= 2:
                distances = np.sum((points[:, None] - points[None, :])**2, axis=2)
                i, j = np.unravel_index(distances.argmax(), distances.shape)
                segments.append(points[[i, j]][:, axes])
    return segments


def render_motion(root):
    """Write initial/final cross sections and cumulative motion of accepted meshes."""
    from surface_mesh.mesh import load_mesh
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.collections import LineCollection
    from matplotlib.lines import Line2D

    root = Path(root)
    history = json.loads((root / 'history.json').read_text())
    sequence = history['accepted_sequence']
    check = history.get('int_inspired', {}).get('enable_safeguards',
                                               history.get('mode') != 'int_inspired')
    meshes = [load_mesh(root / name, validate_geometry=check) for name in sequence]
    initial, final = meshes[0], meshes[-1]
    # New handles may reuse old integers; once remeshed, row identities are gone.
    remeshed_names = {trial['mesh'] for trial in history.get('trials', []) if 'remeshing' in trial}
    remeshed = False
    records = []
    for name, mesh in zip(sequence, meshes):
        remeshed = (remeshed or name in remeshed_names
                    or not np.array_equal(mesh.vertex_handles, initial.vertex_handles)
                    or not np.array_equal(mesh.faces, initial.faces))
        if remeshed:
            from gpytoolbox import squared_distance
            forward = squared_distance(mesh.coordinates, initial.coordinates, initial.faces, use_cpp=True)[0]
            reverse = squared_distance(initial.coordinates, mesh.coordinates, mesh.faces, use_cpp=True)[0]
            records.append(dict(max_motion_mm=None, max_motion_without_translation_mm=None,
                                max_sampled_surface_distance_mm=float(
                                    10 * np.sqrt(max(forward.max(), reverse.max())))))
            continue
        delta = mesh.coordinates - initial.coordinates
        records.append(dict(
            max_motion_mm=float(10 * np.linalg.norm(delta, axis=1).max()),
            max_motion_without_translation_mm=float(
                10 * np.linalg.norm(delta - delta.mean(axis=0), axis=1).max())))
    figure = Figure(figsize=(10, 9), layout='constrained')
    FigureCanvasAgg(figure)
    axes = figure.subplots(2, 2).ravel()
    center = (initial.coordinates.min(axis=0) + initial.coordinates.max(axis=0)) / 2
    lower = np.minimum(initial.coordinates.min(0), final.coordinates.min(0))
    upper = np.maximum(initial.coordinates.max(0), final.coordinates.max(0))
    width = float((upper - lower).max()) * 1.12
    for axis, normal in zip(axes[:3], (2, 1, 0)):
        shown = [k for k in range(3) if k != normal]
        for mesh, color, style in ((initial, '#666666', '--'), (final, '#0066bb', '-')):
            axis.add_collection(LineCollection(
                section_segments(mesh.coordinates, mesh.faces, normal, center[normal]),
                colors=color, linestyles=style, linewidths=1.4))
        for k, setter in zip(shown, (axis.set_xlim, axis.set_ylim)):
            midpoint = (lower[k] + upper[k]) / 2
            setter(midpoint - width / 2, midpoint + width / 2)
        axis.set_aspect('equal')
        axis.set_xlabel(f'{"xyz"[shown[0]]} [cm]')
        axis.set_ylabel(f'{"xyz"[shown[1]]} [cm]')
        axis.set_title(f'{"xyz"[normal]} = {center[normal]:.3g} cm (fixed slice)')
        axis.legend(handles=[Line2D([], [], color='#666666', ls='--', label='Initial'),
                             Line2D([], [], color='#0066bb', label=f'Update {len(meshes)-1}')])
    axis = axes[3]
    for key, label in (('max_motion_mm', 'Maximum displacement'),
                       ('max_motion_without_translation_mm', 'After subtracting mean translation')):
        axis.plot(range(len(records)), [record[key] for record in records], 'o-', label=label)
    if remeshed:
        axis.plot(range(len(records)), [r.get('max_sampled_surface_distance_mm', 0. if i == 0 else None)
                                       for i, r in enumerate(records)], 's-',
                  label='Symmetric surface distance sampled at vertices')
    axis.set_xlabel('Accepted outer update')
    axis.set_ylabel('Distance from initial mesh [mm]')
    axis.set_title('Motion measured from actual H5M coordinates')
    axis.legend(fontsize=8)
    axis.grid(alpha=.2)
    figure.suptitle('Mesh close-up: actual geometry, no displacement magnification')
    figure.savefig(root / 'mesh_motion.png', dpi=160)
    figure.clear()
    (root / 'mesh_motion.json').write_text(json.dumps(
        dict(meshes=sequence, motion=records, slice_center_cm=center.tolist()), indent=2) + '\n')
    return records[-1]
