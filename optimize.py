import argparse
import os
from collections import namedtuple
from functools import partial
from pathlib import Path
import tempfile

import numpy as np
import openmc
import openmc.lib
from scipy.optimize import minimize, basinhopping

import geometry_plots
from model import main as generate_model
from config import (
    FLUX_TRIGGER_REL_ERR, TRIGGER_MAX_BATCHES_FACTOR, TRIGGER_BATCH_INTERVAL,
    PARAMETER_CHANGE_TOL_CM, PARAMETER_STABLE_ITERATIONS, SURFACE_PARAMETER_NAME,
    OPTIMIZATION_PARAMETER_NAMES, OPTIMIZATION_METHOD,
    BASIN_HOP_ITERATIONS, BASIN_HOP_STEP_FRACTION,
    BASIN_HOP_TEMPERATURE, BASIN_HOP_SEED,
)


model = None
model_surfaces = None


def initialize_model(parameter_names=None):
    """Generate a fresh model before starting an optimization or sweep."""
    global model, model_surfaces
    model = generate_model(parameter_names=parameter_names)
    model_surfaces = model.geometry.get_all_surfaces()

runtime_args = ['-s', '10']

SimulationSample = namedtuple(
    'SimulationSample',
    ['parameter', 'tally_mean', 'tally_std_dev', 'geometry_derivative', 'dJ', 'datt'],
)

SIMULATION_SAMPLE_DTYPE = np.dtype([
    ('parameter', 'f8'),
    ('tally_mean', 'f8'),
    ('tally_std_dev', 'f8'),
    ('geometry_derivative', 'f8'),
    ('dJ', 'f8'),
    ('datt', 'f8'),
])

def save_history(samples, output_path, parameter_names=None):
    """Replace the NumPy history atomically so completed evaluations survive."""
    output_path = Path(output_path)
    if output_path.suffix != '.npy':
        output_path = Path(str(output_path) + '.npy')
    dtype = SIMULATION_SAMPLE_DTYPE
    if parameter_names is not None:
        count = len(parameter_names)
        dtype = np.dtype([
            (name, 'f8', (count,)) if count > 1 and name in ('parameter', 'geometry_derivative', 'dJ', 'datt')
            else (name, 'f8') for name in SIMULATION_SAMPLE_DTYPE.names
        ])
        # Store coordinate names alongside the vectors in the .npy file.
        dtype = np.dtype(dtype.descr + [('parameter_names', 'U16', (count,))])
        samples = [
            tuple(float(np.asarray(value).reshape(-1)[0])
                  if count == 1 and name in ('parameter', 'geometry_derivative', 'dJ', 'datt')
                  else value for name, value in zip(SIMULATION_SAMPLE_DTYPE.names, sample))
            + (tuple(parameter_names),) for sample in samples
        ]
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(dir=output_path.parent, suffix='.npy',
                                         delete=False) as stream:
            temporary_path = Path(stream.name)
            np.save(stream, np.array(samples, dtype=dtype))
        os.replace(temporary_path, output_path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


class ParameterConvergence:
    """Stop after the configured number of small changes in accepted iterates."""

    def __init__(self, initial_parameter):
        self.previous_value = np.atleast_1d(initial_parameter).copy()
        self.stable_iterations = 0
        self.stopped = False

    def __call__(self, intermediate_result):
        parameter_value = np.asarray(intermediate_result.x)
        change = np.max(np.abs(parameter_value - self.previous_value))
        self.previous_value = parameter_value.copy()
        self.stable_iterations = (
            self.stable_iterations + 1 if change <= PARAMETER_CHANGE_TOL_CM else 0
        )
        print(f'Accepted parameters: {parameter_value} cm; maximum change: {change:.6g} cm; '
              f'stable iterations: {self.stable_iterations}/{PARAMETER_STABLE_ITERATIONS}')
        if self.stable_iterations >= PARAMETER_STABLE_ITERATIONS:
            self.stopped = True
            raise StopIteration

def configure_tally_trigger():
    """Apply a relative-error trigger to the flux tally."""
    flux_tally = next(tally for tally in model.tallies if tally.id == 1)
    trigger = openmc.Trigger('rel_err', FLUX_TRIGGER_REL_ERR)
    trigger.scores = ['flux']
    flux_tally.triggers = [trigger]

    model.settings.trigger_active = True
    model.settings.trigger_max_batches = (
        model.settings.batches * TRIGGER_MAX_BATCHES_FACTOR
    )
    model.settings.trigger_batch_interval = TRIGGER_BATCH_INTERVAL

def parameter_plot_extent(max_parameter):
    return geometry_plots.parameter_plot_extent(
        model, max_parameter, min_parameter=get_constraints()[0],
    )


def optimization_plot_extent(parameter_names, bounds):
    from itertools import product
    original = get_parameters(parameter_names)
    boxes = []
    try:
        for corner in product(*bounds):
            for name, value in zip(parameter_names, corner):
                model_surfaces[1].coefficients[name] = float(value)
            boxes.extend(cell.bounding_box for cell in model.geometry.get_all_cells().values()
                         if cell.fill_type == 'material')
    finally:
        for name, value in zip(parameter_names, original):
            model_surfaces[1].coefficients[name] = float(value)
    lower = np.min([box.lower_left for box in boxes], axis=0)
    upper = np.max([box.upper_right for box in boxes], axis=0)
    if not np.all(np.isfinite([lower, upper])):
        raise ValueError('Material geometry must have finite bounds for optimization plots.')
    return tuple((lower + upper) / 2), tuple(1.1 * (upper - lower)[:2])


def plot_current_parameter(plot_dir, run_index, plot_extent):
    return geometry_plots.plot_current_parameter(model, plot_dir, run_index, plot_extent)


def evaluate_current_parameter(plot_dir=None, run_index=0, plot_extent=None,
                               parameter_names=None):
    """Return named results for the current geometry parameter value."""
    names = (SURFACE_PARAMETER_NAME,) if parameter_names is None else parameter_names
    parameter_value = get_parameters(names)
    if plot_dir is not None and plot_extent is None:
        plot_extent = (parameter_plot_extent(get_constraints()[1]) if parameter_names is None
                       else optimization_plot_extent(names, [get_constraints(name) for name in names]))
    with openmc.lib.run_in_memory(args=runtime_args, output=False):
        if plot_dir is not None:
            plot_path = geometry_plots.plot_current_parameter(
                model, plot_dir, run_index, plot_extent,
                parameter_names=names if parameter_names is not None else None,
            )
            print('Saved geometry plot to:', plot_path.resolve())
        openmc.lib.simulation_init()
        tally = openmc.lib.tallies[1]
        geom_derivative = openmc.lib.geometry_derivatives[1]
        try:
            openmc.lib.run(output=False)
            num_realizations = tally.num_realizations
            rows = [geom_derivative.surface_parameters.index(name) for name in names]
            normalization = max(num_realizations, 1)
            def components(values):
                values = np.asarray(values)
                if values.ndim != 2 or values.shape[1] != 1:
                    raise ValueError('Optimization requires a single derivative tally bin.')
                result = values[rows, 0].copy()
                return float(result[0]) if parameter_names is None else result

            sample = SimulationSample(
                parameter=float(parameter_value[0]) if parameter_names is None else parameter_value.copy(),
                tally_mean=float(tally.mean.flat[-1]),
                tally_std_dev=float(tally.std_dev.flat[-1]),
                geometry_derivative=components(geom_derivative.mean),
                dJ=components(geom_derivative.dJ / normalization),
                datt=components(geom_derivative.datt / normalization),
            )
            if sample.tally_mean == 0.0:
                flux_rel_err = np.inf
            else:
                flux_rel_err = sample.tally_std_dev / abs(sample.tally_mean)
            print("Tally mean:", sample.tally_mean)
            print("Tally std. dev.:", sample.tally_std_dev)
            print("Tally rel. err.:", flux_rel_err)
            print("Tally realizations:", tally.num_realizations)
            print("Geometric derivative mean:", sample.geometry_derivative)
            print("Geometric derivative dJ mean:", sample.dJ)
            print("Geometric derivative datt mean:", sample.datt)
        finally:
            openmc.lib.simulation_finalize()
        return sample

def run_simulation(parameter_names=None):
    """Return the water flux mean and its derivative for the current geometry."""
    sample = evaluate_current_parameter(parameter_names=parameter_names)
    return sample.tally_mean, sample.geometry_derivative

def get_constraints(parameter_name=SURFACE_PARAMETER_NAME):
    """Return the lower and upper bounds for the optimized geometry parameter."""
    water_sphere = model_surfaces[2]
    iron_sphere = model_surfaces[1]
    if parameter_name == 'r':
        return 0.01, water_sphere.x0 - water_sphere.r
    if parameter_name == 'x0':
        water_sphere = model_surfaces[2]
        return -10, 10
    if parameter_name == 'y0':
        water_sphere = model_surfaces[2]
        return -10, 10
    if parameter_name == 'z0':
        water_sphere = model_surfaces[2]
        return -water_sphere.r - iron_sphere.r, water_sphere.r + iron_sphere.r

def validate_parameter_names(parameter_names):
    names = tuple(parameter_names)
    if not names or len(set(names)) != len(names):
        raise ValueError('Optimization parameter names must be nonempty and unique.')
    if any(name not in ('r', 'x0', 'y0', 'z0') for name in names):
        raise ValueError('Supported iron sphere parameters are r, x0, y0, and z0.')
    return names


def get_parameters(parameter_names):
    return np.array([model_surfaces[1].coefficients[name] for name in parameter_names])


def update_parameters(parameters, parameter_names):
    """Update every optimized coordinate and export once per evaluation."""
    names = validate_parameter_names(parameter_names)
    values = np.asarray(parameters, dtype=float)
    if values.shape != (len(names),) or not np.all(np.isfinite(values)):
        raise ValueError('Expected one finite value per optimization parameter.')
    for name, value in zip(names, values):
        model_surfaces[1].coefficients[name] = float(value)
    configure_tally_trigger()
    model.export_to_model_xml()


def get_parameter():
    """Gather the current geometry parameter, the configured coefficient of the iron sphere."""
    iron_sphere = model_surfaces[1]
    return iron_sphere.coefficients[SURFACE_PARAMETER_NAME]

def update_parameter(new_parameter):
    """Update the optimized geometry parameter and export it for simulation."""
    print("Updating iron sphere parameter to:", new_parameter)
    iron_sphere = model_surfaces[1]
    iron_sphere.coefficients[SURFACE_PARAMETER_NAME] = new_parameter
    configure_tally_trigger()
    model.export_to_model_xml()

def flux_objective(parameter, samples=None, plot_dir=None, plot_extent=None,
                   output_path=None, use_jac=True, parameter_names=None):
    """Return the flux mean, optionally with its computed gradient."""
    names = validate_parameter_names(OPTIMIZATION_PARAMETER_NAMES if parameter_names is None else parameter_names)
    update_parameters(parameter, names)
    sample = evaluate_current_parameter(
        plot_dir=plot_dir,
        run_index=len(samples) if samples is not None else 0,
        plot_extent=plot_extent, parameter_names=names,
    )
    if samples is not None:
        samples.append(sample)
        if output_path is not None:
            save_history(samples, output_path, parameter_names=names)
    objective = sample.tally_mean
    if not use_jac:
        return objective
    gradient = np.asarray(sample.geometry_derivative, dtype=float)
    if gradient.shape != (len(names),) or not np.all(np.isfinite(gradient)):
        raise ValueError("Expected one finite gradient component per parameter.")
    return objective, gradient

class BoundedBasinStep:
    """Jump in all coordinates, scaling distances by their bound widths."""

    def __init__(self, bounds, step_fraction, rng):
        self.lower, self.upper = np.asarray(bounds, dtype=float).T
        self.width = self.upper - self.lower
        if not np.all(np.isfinite([self.lower, self.upper])) or np.any(self.width <= 0):
            raise ValueError('Basin-hopping requires finite bounds with positive widths.')
        self.stepsize = step_fraction
        self.rng = rng

    def __call__(self, values):
        proposal = values + self.rng.uniform(-1., 1., size=len(values)) * self.stepsize * self.width
        # Reflect overshoots instead of accumulating proposals at a bound.
        offset = (proposal - self.lower) % (2 * self.width)
        values[:] = self.lower + np.minimum(offset, 2 * self.width - offset)
        return values


def optimize_parameter(output_path='optimization_history.npy', plot_dir=None,
                       use_jac=True, parameter_names=None, method=OPTIMIZATION_METHOD,
                       hop_iterations=BASIN_HOP_ITERATIONS,
                       hop_step_fraction=BASIN_HOP_STEP_FRACTION,
                       hop_temperature=BASIN_HOP_TEMPERATURE, hop_seed=BASIN_HOP_SEED):
    """Minimize water flux and save every objective evaluation in run order."""
    if method not in ('local', 'basinhopping'):
        raise ValueError('Optimization method must be local or basinhopping.')
    if (not isinstance(hop_iterations, (int, np.integer)) or hop_iterations < 0
            or not np.isfinite(hop_step_fraction) or hop_step_fraction <= 0
            or not np.isfinite(hop_temperature) or hop_temperature < 0):
        raise ValueError('Hops must be a nonnegative integer, step fraction positive, and temperature nonnegative.')
    names = validate_parameter_names(OPTIMIZATION_PARAMETER_NAMES if parameter_names is None else parameter_names)
    initialize_model(names)
    initial_value = get_parameters(names)
    print("Initial iron sphere parameter:", initial_value)
    bounds = [get_constraints(name) for name in names]
    print("Optimization bounds:", bounds)
    samples = []
    convergence = ParameterConvergence(initial_value)
    plot_extent = optimization_plot_extent(names, bounds) if plot_dir is not None else None
    objective = partial(
        flux_objective, samples=samples, plot_dir=plot_dir, plot_extent=plot_extent,
        output_path=output_path, use_jac=use_jac, parameter_names=names,
    )
    local_options = dict(method='L-BFGS-B', jac=True if use_jac else None, bounds=bounds)
    if method == 'local':
        result = minimize(objective, initial_value, callback=convergence, **local_options)
        if convergence.stopped:
            result.success = True
            result.status = 0
            result.message = (f'All parameters {names} changed by at most '
                              f'{PARAMETER_CHANGE_TOL_CM:g} cm for '
                              f'{PARAMETER_STABLE_ITERATIONS} consecutive accepted iterations.')
    else:
        rng = np.random.default_rng(hop_seed)
        step = BoundedBasinStep(bounds, hop_step_fraction, rng)
        basin_count = 0

        def report_basin(values, value, accepted):
            nonlocal basin_count
            coordinates = dict(zip(names, values))
            print(f'Basin {basin_count}: parameters={coordinates}; flux={value:.10g}; accepted={accepted}')
            basin_count += 1

        # Local convergence must not terminate global exploration or leave a
        # persistent StopIteration flag shared between separate basin searches.
        result = basinhopping(
            objective, initial_value, niter=hop_iterations, T=hop_temperature,
            take_step=step, minimizer_kwargs=local_options, callback=report_basin,
            rng=rng,
        )
        result.success = result.lowest_optimization_result.success
        result.jac = result.lowest_optimization_result.jac
    print('Optimization method:', method)
    print('Saved optimization history to:', output_path)
    update_parameters(result.x, names)
    return result

def collect_parameter_sweep(num_values=100, output_path='parameter_sweep.npy',
                            plot_dir=None):
    """Sample the parameter range; optionally save material slices to plot_dir."""
    initialize_model((SURFACE_PARAMETER_NAME,))
    initial_parameter = get_parameter()
    lower_bound, upper_bound = get_constraints()
    parameters = np.linspace(lower_bound, upper_bound, num_values, endpoint=True)
    plot_extent = parameter_plot_extent(upper_bound) if plot_dir is not None else None
    samples = []

    try:
        for run_index, parameter in enumerate(parameters):
            update_parameter(float(parameter))
            samples.append(evaluate_current_parameter(
                plot_dir=plot_dir, run_index=run_index, plot_extent=plot_extent,
            ))
            save_history(samples, output_path)
    finally:
        update_parameter(initial_parameter)

    results = np.array(samples, dtype=SIMULATION_SAMPLE_DTYPE)
    print("Saved parameter sweep results to:", output_path)
    return results

def main():
    parser = argparse.ArgumentParser(description='Optimize or sweep the iron sphere surface parameter.')
    parser.add_argument('--sweep', action='store_true', help='Evaluate a parameter sweep.')
    parser.add_argument('-n', '--num-values', type=int, default=100)
    parser.add_argument('-o', '--output', help='NumPy output file (default: optimization_history.npy, or parameter_sweep.npy with --sweep).')
    parser.add_argument('--plot-dir', help='Save a geometry plot for each evaluation here.')
    parser.add_argument('--no-jac', action='store_true',
                        help='Let L-BFGS-B estimate gradients numerically (jac=None).')
    parser.add_argument('--method', choices=('basinhopping', 'local'), default=OPTIMIZATION_METHOD,
                        help='Optimization strategy (default: %(default)s).')
    parser.add_argument('--hops', type=int, default=BASIN_HOP_ITERATIONS,
                        help='Number of basin jumps after initial local minimization.')
    parser.add_argument('--hop-step-fraction', type=float, default=BASIN_HOP_STEP_FRACTION,
                        help='Jump scale as a fraction of each coordinate bound width.')
    parser.add_argument('--hop-temperature', type=float, default=BASIN_HOP_TEMPERATURE,
                        help='Temperature for accepting uphill basins, in objective units.')
    parser.add_argument('--hop-seed', type=int, default=BASIN_HOP_SEED,
                        help='Random seed for reproducible basin jumps.')
    args = parser.parse_args()
    if args.sweep:
        collect_parameter_sweep(args.num_values, args.output or 'parameter_sweep.npy',
                                plot_dir=args.plot_dir)
        return
    result = optimize_parameter(args.output or 'optimization_history.npy',
                                plot_dir=args.plot_dir, use_jac=not args.no_jac,
                                method=args.method, hop_iterations=args.hops,
                                hop_step_fraction=args.hop_step_fraction,
                                hop_temperature=args.hop_temperature, hop_seed=args.hop_seed)
    print("Optimization success:", result.success)
    print("Optimized parameters:", dict(zip(OPTIMIZATION_PARAMETER_NAMES, result.x)))
    print("Optimized flux:", result.fun)
    print("Final gradient:", result.jac)
    print("Termination message:", result.message)


if __name__ == '__main__':
    main()
