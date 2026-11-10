"""Capture regular-mesh tallies and display saved XY depth averages."""

import numpy as np


def capture_mesh_tally(model):
    """Copy the mesh_flux mean and mesh definition before library finalization."""
    import openmc
    import openmc.lib

    tally = next((tally for tally in model.tallies if tally.name == 'mesh_flux'), None)
    if tally is None:
        return None
    if (len(tally.filters) != 1 or type(tally.filters[0]) is not openmc.MeshFilter
            or not isinstance(tally.filters[0].mesh, openmc.RegularMesh)
            or len(tally.scores) != 1 or tally.nuclides not in ([], ['total'])):
        raise ValueError('mesh_flux must have one regular mesh filter and one total score.')
    mesh = tally.filters[0].mesh
    if len(mesh.dimension) != 3:
        raise ValueError('mesh_flux must use a three-dimensional regular mesh.')
    # OpenMC bins vary fastest in x, then y, then z.
    mean = openmc.lib.tallies[tally.id].mean.reshape(mesh.dimension, order='F').copy()
    return {'mean': mean, 'lower_left': np.array(mesh.lower_left),
            'upper_right': np.array(mesh.upper_right), 'score': tally.scores[0]}


def mesh_history_fields(snapshots):
    """Return non-object dtype fields and rows for a consistent mesh history."""
    if not any(snapshot is not None for snapshot in snapshots):
        return [], [() for _ in snapshots]
    if any(snapshot is None for snapshot in snapshots):
        raise ValueError('Every history sample must have mesh data when saving a mesh tally.')
    first = snapshots[0]
    shape = first['mean'].shape
    if len(shape) != 3 or min(shape) == 0:
        raise ValueError('Mesh means must be nonempty three-dimensional arrays.')
    lower, upper = first['lower_left'], first['upper_right']
    if (np.shape(lower) != (3,) or np.shape(upper) != (3,)
            or not np.all(np.isfinite([lower, upper])) or np.any(upper <= lower)):
        raise ValueError('Mesh bounds must be finite and have positive widths.')
    for snapshot in snapshots:
        if (snapshot['mean'].shape != shape
                or not np.array_equal(snapshot['lower_left'], lower)
                or not np.array_equal(snapshot['upper_right'], upper)
                or snapshot['score'] != first['score']):
            raise ValueError('Mesh dimensions, bounds, and score must match across runs.')
        if not np.all(np.isfinite(snapshot['mean'])):
            raise ValueError('Mesh tally means must be finite.')
    fields = [('mesh_mean', 'f8', shape), ('mesh_lower_left', 'f8', (3,)),
              ('mesh_upper_right', 'f8', (3,)),
              ('mesh_score', f"U{len(first['score'])}")]
    rows = [(s['mean'], s['lower_left'], s['upper_right'], s['score']) for s in snapshots]
    return fields, rows


def add_mesh_history_plot(fig, subplot_spec, results, axis=None):
    """Draw volume-normalized XY means with one logarithmic scale for all runs."""
    from matplotlib.colors import LogNorm

    mean = results['mesh_mean']
    lower, upper = results['mesh_lower_left'][0], results['mesh_upper_right'][0]
    score = str(results['mesh_score'][0])
    # Sum the voxel-integrated tally through z, then divide by column volume.
    # This is the depth average of the volumetric score, not a plane slice.
    widths = (upper - lower) / np.array(mean.shape[1:])
    column_volume = widths[0] * widths[1] * (upper[2] - lower[2])
    frames = (mean.sum(axis=3) / column_volume).transpose(0, 2, 1)
    frames = np.ma.masked_where(~np.isfinite(frames) | (frames <= 0), frames)
    positive = frames.compressed()
    minimum, maximum = (float(positive.min()), float(positive.max())) if positive.size else (1., 10.)
    if minimum == maximum:
        minimum, maximum = minimum / np.sqrt(10.), maximum * np.sqrt(10.)
    overlay = axis is not None
    if axis is None:
        axis = fig.add_subplot(subplot_spec)
    image = axis.imshow(
        frames[0], origin='lower', interpolation='nearest', cmap='viridis',
        extent=(lower[0], upper[0], lower[1], upper[1]),
        norm=LogNorm(vmin=minimum, vmax=maximum), aspect='equal', zorder=0,
    )
    axis.set_xlabel('x [cm]')
    axis.set_ylabel('y [cm]')
    axis.set_title(f'Mesh {score}: depth average, z = {lower[2]:g}–{upper[2]:g} cm')
    units = {'flux': 'cm⁻² per source particle',
             'heating': 'eV cm⁻³ per source particle',
             'heating-local': 'eV cm⁻³ per source particle'}.get(
                 score, 'cm⁻³ per source particle')
    label = f'{score} [{units}]'
    if overlay:
        label += f'\nDepth average, z = {lower[2]:g}–{upper[2]:g} cm'
    fig.colorbar(image, ax=axis, label=label, shrink=0.8)
    return image, frames
