"""Select an explicit model variant while keeping the original model intact."""

import importlib
from pathlib import Path


def objective_tally(model):
    """Validate the scalar objective and its derivative used by the worker."""
    tally = next((t for t in model.tallies if t.id == 1), None)
    if (tally is None or tally.num_filter_bins != 1 or len(tally.scores) != 1
            or len(tally.nuclides) > 1):
        raise ValueError('Optimization requires tally 1 with exactly one objective bin')
    derivative = next((d for d in model.geometry_derivatives if d.id == 1), None)
    if derivative is None or derivative.tally is not tally:
        raise ValueError('Geometric derivative 1 must reference objective tally 1')
    return tally


def model_module(settings):
    name = {'sphere': 'model', 'box': 'model_box'}[settings.model_variant]
    return importlib.import_module(f'.{name}', __package__)


def model_source_path(settings):
    return Path(model_module(settings).__file__).resolve()


def source_position_in_mesh(settings):
    """Express the selected model's point source in translated mesh coordinates."""
    import numpy as np
    module = model_module(settings)
    return np.asarray(module.SOURCE_CENTER_CM) - np.asarray(module.IRON_CENTER_CM)


def build_model(settings, mesh_path, *, seed=None, budget=1):
    kwargs = dict(particles=settings.particles * budget, batches=settings.batches,
                  seed=settings.seed if seed is None else seed)
    if settings.model_variant == 'box':
        kwargs.update(container_lower=settings.box_container_lower,
                      container_upper=settings.box_container_upper)
    else:
        kwargs['container_radius'] = settings.container_radius
    return model_module(settings).build_model(mesh_path, **kwargs)
