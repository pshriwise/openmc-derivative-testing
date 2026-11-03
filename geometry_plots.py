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
                             parameter_name=SURFACE_PARAMETER_NAME, parameter_names=None,
                             return_extent=False):
    """Render geometry, optionally returning transparent pixels and XY extents.

    The returned extents describe the full canvas and the plot area, respectively,
    in physical coordinates, so a mesh can be placed beneath the saved image.
    """
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
        if return_extent:
            # The renderer assigns white to void and colors to all materials.
            # Remove only the void fill, preserving geometry, source, and labels.
            artist = axis.images[0]
            rgb = np.asarray(artist.get_array())[..., :3]
            if rgb.dtype == np.uint8:
                rgb = rgb / 255.
            rgba = np.concatenate((rgb, (~np.all(rgb == 1., axis=-1))[..., None]), axis=-1)
            artist.set_data(rgba)
            fig.patch.set_alpha(0)
            axis.patch.set_alpha(0)
        canvas = FigureCanvasAgg(fig)
        canvas.draw()
        image = np.asarray(canvas.buffer_rgba()).copy()
        if return_extent:
            height, width = image.shape[:2]
            lower, upper = axis.transData.inverted().transform([(0, 0), (width, height)])
            image_extent = np.array([lower[0], upper[0], lower[1], upper[1]])
            return image, image_extent, np.array([*axis.get_xlim(), *axis.get_ylim()])
        return image
    finally:
        plt.close(fig)


def legacy_plot_extent(results):
    """Recover the CSG view from saved run settings without running transport.

    Histories predating explicit image extents use the fixed geometry and source
    in model.py, with the saved initial iron coefficients and parameter bounds.
    """
    from itertools import product
    import warnings

    import matplotlib.pyplot as plt
    import openmc
    from model import build_model
    from plot_parameter_sweep import history_metadata

    metadata = history_metadata(results)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', openmc.IDWarning)
        model = build_model(iron=metadata['initial_iron'])
    bounds = np.asarray(metadata['bounds'])
    names = tuple(results['parameter_names'][0])
    if bounds.ndim == 1:
        origin, width = parameter_plot_extent(
            model, bounds[1], parameter_name=names[0], min_parameter=bounds[0],
        )
    else:
        sphere = model.geometry.get_all_surfaces()[1]
        boxes = []
        for corner in product(*bounds):
            for name, value in zip(names, corner):
                sphere.coefficients[name] = float(value)
            boxes.extend(cell.bounding_box for cell in model.geometry.get_all_cells().values()
                         if cell.fill_type == 'material')
        lower = np.min([box.lower_left for box in boxes], axis=0)
        upper = np.max([box.upper_right for box in boxes], axis=0)
        origin, width = (lower + upper) / 2, 1.1 * (upper - lower)[:2]
    # Reproduce imshow/scatter autoscaling: a point source may extend the view
    # beyond the material bounding box. This needs no geometry rasterization.
    fig, axis = plt.subplots()
    try:
        axis.imshow(np.zeros((2, 2)), extent=(origin[0] - width[0] / 2,
                    origin[0] + width[0] / 2, origin[1] - width[1] / 2,
                    origin[1] + width[1] / 2))
        for source in model.settings.source:
            if (isinstance(source, openmc.IndependentSource)
                    and isinstance(source.space, openmc.stats.Point)
                    and abs(source.space.xyz[2] - origin[2]) < 1.):
                axis.scatter(*source.space.xyz[:2])
        return np.array([*axis.get_xlim(), *axis.get_ylim()])
    finally:
        plt.close(fig)


def image_extent_from_frame(image, plot_extent):
    """Locate the saved black axes frame and map canvas pixels to XY coordinates."""
    dark = (image[..., :3].max(axis=2) < 64) & (image[..., 3] > 0)
    edges = []
    for dimension in (0, 1):
        counts = dark.sum(axis=dimension)
        positions = np.flatnonzero(counts > 0.8 * counts.max())
        bands = np.split(positions, np.flatnonzero(np.diff(positions) > 1) + 1)
        if len(bands) != 2:
            raise ValueError('Could not locate the axes frame in the saved geometry image.')
        edges.append([band.mean() + 0.5 for band in bands])
    left, right = edges[0]
    top, bottom = edges[1]
    height, width = image.shape[:2]
    xmin, xmax, ymin, ymax = plot_extent
    dx = (xmax - xmin) / (right - left)
    dy = (ymax - ymin) / (bottom - top)
    return np.array([xmin - left * dx, xmax + (width - right) * dx,
                     ymin - (height - bottom) * dy, ymax + top * dy])


class HistoryGeometryPlot:
    """Display saved geometry over an optional, physically aligned mesh layer."""

    def __init__(self, fig, subplot_spec, results, images, show_mesh=False):
        from matplotlib.patches import Rectangle
        from mesh_plots import add_mesh_history_plot

        self.results = results
        self.images = images
        self.legacy = show_mesh and not {'image_extent', 'plot_extent'} <= set(results.dtype.names)
        if self.legacy:
            self.plot_extent = legacy_plot_extent(results)
        self.axis = fig.add_subplot(subplot_spec)
        self.mesh_image = None
        if show_mesh:
            self.mesh_image, self.mesh_frames = add_mesh_history_plot(
                fig, subplot_spec, results, axis=self.axis,
            )
            self.clip = Rectangle((0, 0), 1, 1, transform=self.axis.transData)
            self.axis.set_title('')
        self.image = self.axis.imshow(images[0], origin='upper', zorder=2)
        self.axis.set_axis_off()
        self.update(0)

    def update(self, index):
        """Update both layers and the saved coordinate mapping for this run."""
        pixels = self.images[index]
        if self.legacy:
            # Old images have opaque white void pixels. Make them transparent
            # at display time; preserve the history file and colored geometry.
            pixels = pixels.copy()
            pixels[np.all(pixels[..., :3] == 255, axis=-1), 3] = 0
        self.image.set_data(pixels)
        if self.mesh_image is None:
            return (self.image,)
        if self.legacy:
            plot_extent = self.plot_extent
            extent = image_extent_from_frame(self.images[index], plot_extent)
        else:
            plot_extent = self.results['plot_extent'][index]
            extent = self.results['image_extent'][index]
        self.image.set_extent(extent)
        self.axis.set_xlim(extent[:2])
        self.axis.set_ylim(extent[2:])
        xmin, xmax, ymin, ymax = plot_extent
        self.clip.set_bounds(xmin, ymin, xmax - xmin, ymax - ymin)
        self.mesh_image.set_clip_path(self.clip)
        self.mesh_image.set_data(self.mesh_frames[index])
        return self.image, self.mesh_image
