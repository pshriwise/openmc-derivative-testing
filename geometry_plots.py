import numpy as np

from config import SURFACE_PARAMETER_NAME, SURFACE_PARAMETER_UNITS


def parameter_plot_extent(model, max_parameter, parameter_name=SURFACE_PARAMETER_NAME,
                          min_parameter=None):
    """Return a fixed XY view enclosing material cells at parameter endpoints."""
    iron_sphere = model.geometry.get_all_surfaces()[1]
    original_parameter = iron_sphere.coefficients[parameter_name]
    bounds = []
    try:
        endpoints = [original_parameter, max_parameter]
        if min_parameter is not None:
            endpoints.append(min_parameter)
        for value in endpoints:
            iron_sphere.coefficients[parameter_name] = float(value)
            bounds.extend(cell.bounding_box for cell in model.geometry.get_all_cells().values()
                          if cell.fill_type == 'material')
        lower = np.min([box.lower_left for box in bounds], axis=0)
        upper = np.max([box.upper_right for box in bounds], axis=0)
    finally:
        iron_sphere.coefficients[parameter_name] = original_parameter
    if not np.all(np.isfinite([lower, upper])):
        raise ValueError('Material geometry must have finite bounds for sweep plots.')
    origin = tuple((lower + upper) / 2.0)
    width = tuple(1.1 * (upper - lower)[:2])
    return origin, width


def render_current_parameter(model, run_index, plot_extent, source_samples=20,
                             parameter_name=SURFACE_PARAMETER_NAME, parameter_names=None):
    """Return a uint8 RGBA material slice with source sites and a run title."""
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    import openmc

    origin, width = plot_extent
    sources = model.settings.source
    point_sources = all(isinstance(source, openmc.IndependentSource)
                        and isinstance(source.space, openmc.stats.Point)
                        for source in sources) and bool(sources)
    source_style = {'marker': 'x', 'color': 'red', 's': 50,
                    'linewidths': 1.5, 'zorder': 3, 'label': 'Source'}
    colors = {material: ('gray' if material.name == 'iron' else 'blue')
              for material in model.geometry.get_all_materials().values()}
    fig, axis = plt.subplots(figsize=(9, 6), dpi=150, constrained_layout=True)
    try:
        model.plot(origin=origin, width=width, pixels=(1200, 800),
                   basis='xy', color_by='material', colors=colors,
                   legend=True, axes=axis,
                   n_samples=0 if point_sources else source_samples,
                   source_kwargs=source_style)
        if point_sources and source_samples:
            # Point-source positions are exact; avoid sampling unused energy
            # and direction distributions just to draw their locations.
            weights = np.array([source.strength for source in sources])
            indices = np.random.default_rng(run_index).choice(
                len(sources), size=source_samples, p=weights / weights.sum(),
            )
            sites = np.array([sources[index].space.xyz for index in indices])
            sites = sites[np.abs(sites[:, 2] - origin[2]) < 1.0]
            axis.scatter(sites[:, 0], sites[:, 1], **source_style)
        if source_samples:
            material_legend = axis.get_legend()
            axis.add_artist(material_legend)
            axis.legend(loc='upper left')
        axis.set_xlabel('x [cm]')
        axis.set_ylabel('y [cm]')
        names = (parameter_name,) if parameter_names is None else parameter_names
        coefficients = model.geometry.get_all_surfaces()[1].coefficients
        label = ', '.join(f'{name} = {coefficients[name]:.8g}' for name in names)
        axis.set_title(f'Run {run_index}: iron sphere {label} {SURFACE_PARAMETER_UNITS}')
        canvas = FigureCanvasAgg(fig)
        canvas.draw()
        return np.asarray(canvas.buffer_rgba()).copy()
    finally:
        plt.close(fig)
