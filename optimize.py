import argparse
import os
from collections import namedtuple
from functools import partial
from pathlib import Path
import tempfile

import numpy as np
import openmc
import openmc.lib
from scipy.optimize import minimize

import geometry_plots
from config import (
    FLUX_TRIGGER_REL_ERR, TRIGGER_MAX_BATCHES_FACTOR, TRIGGER_BATCH_INTERVAL,
    PARAMETER_CHANGE_TOL_CM, PARAMETER_STABLE_ITERATIONS, SURFACE_PARAMETER_NAME,
)


model = openmc.Model.from_model_xml('model.xml')
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

def save_history(samples, output_path):
    """Replace the NumPy history atomically so completed evaluations survive."""
    output_path = Path(output_path)
    if output_path.suffix != '.npy':
        output_path = Path(str(output_path) + '.npy')
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(dir=output_path.parent, suffix='.npy',
                                         delete=False) as stream:
            temporary_path = Path(stream.name)
            np.save(stream, np.array(samples, dtype=SIMULATION_SAMPLE_DTYPE))
        os.replace(temporary_path, output_path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


class ParameterConvergence:
    """Stop after the configured number of small changes in accepted iterates."""

    def __init__(self, initial_parameter):
        self.previous_value = initial_parameter
        self.stable_iterations = 0
        self.stopped = False

    def __call__(self, intermediate_result):
        parameter_value = float(intermediate_result.x[0])
        change = abs(parameter_value - self.previous_value)
        self.previous_value = parameter_value
        self.stable_iterations = (
            self.stable_iterations + 1 if change <= PARAMETER_CHANGE_TOL_CM else 0
        )
        print(f'Accepted parameter: {SURFACE_PARAMETER_NAME}={parameter_value:.10g} cm; change: {change:.6g} cm; '
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


def plot_current_parameter(plot_dir, run_index, plot_extent):
    return geometry_plots.plot_current_parameter(model, plot_dir, run_index, plot_extent)


def evaluate_current_parameter(plot_dir=None, run_index=0, plot_extent=None):
    """Return named results for the current geometry parameter value."""
    parameter_value = get_parameter()
    if plot_dir is not None and plot_extent is None:
        plot_extent = parameter_plot_extent(get_constraints()[1])
    with openmc.lib.run_in_memory(args=runtime_args, output=False):
        if plot_dir is not None:
            plot_path = plot_current_parameter(plot_dir, run_index, plot_extent)
            print('Saved geometry plot to:', plot_path.resolve())
        openmc.lib.simulation_init()
        tally = openmc.lib.tallies[1]
        geom_derivative = openmc.lib.geometry_derivatives[1]
        try:
            openmc.lib.run(output=False)
            num_realizations = tally.num_realizations
            if num_realizations > 0:
                dJ = geom_derivative.dJ.flat[0] / num_realizations
                datt = geom_derivative.datt.flat[0] / num_realizations
            else:
                dJ = geom_derivative.dJ.flat[0]
                datt = geom_derivative.datt.flat[0]

            sample = SimulationSample(
                parameter=float(parameter_value),
                tally_mean=float(tally.mean.flat[-1]),
                tally_std_dev=float(tally.std_dev.flat[-1]),
                geometry_derivative=float(geom_derivative.mean.flat[0]),
                dJ=float(dJ),
                datt=float(datt),
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

def run_simulation():
    """Return the water flux mean and its derivative for the current geometry."""
    sample = evaluate_current_parameter()
    return sample.tally_mean, sample.geometry_derivative

def get_constraints():
    """Return the lower and upper bounds for the optimized geometry parameter."""
    water_sphere = model_surfaces[2]
    iron_sphere = model_surfaces[1]
    if SURFACE_PARAMETER_NAME == 'r':
        return 0.01, water_sphere.x0 - water_sphere.r
    if SURFACE_PARAMETER_NAME == 'x0':
        water_sphere = model_surfaces[2]
        return -water_sphere.r, water_sphere.r
    if SURFACE_PARAMETER_NAME == 'y0':
        water_sphere = model_surfaces[2]
        return -water_sphere.r - iron_sphere.r, water_sphere.r + iron_sphere.r
    if SURFACE_PARAMETER_NAME == 'z0':
        water_sphere = model_surfaces[2]
        return -water_sphere.r - iron_sphere.r, water_sphere.r + iron_sphere.r

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
                   output_path=None, use_jac=True):
    """Return the flux mean, optionally with its computed gradient."""
    parameter = float(parameter[0])
    update_parameter(parameter)
    sample = evaluate_current_parameter(
        plot_dir=plot_dir,
        run_index=len(samples) if samples is not None else 0,
        plot_extent=plot_extent,
    )
    if samples is not None:
        samples.append(sample)
        if output_path is not None:
            save_history(samples, output_path)
    objective = sample.tally_mean
    if not use_jac:
        return objective
    gradient = [sample.geometry_derivative]
    return objective, gradient

def optimize_parameter(output_path='optimization_history.npy', plot_dir=None,
                       use_jac=True):
    """Minimize water flux and save every objective evaluation in run order."""
    initial_value = get_parameter()
    print("Initial iron sphere parameter:", initial_value)
    lower_bound, upper_bound = get_constraints()
    print("Optimization bounds:", lower_bound, upper_bound)
    samples = []
    convergence = ParameterConvergence(initial_value)
    plot_extent = parameter_plot_extent(upper_bound) if plot_dir is not None else None
    result = minimize(
        partial(flux_objective, samples=samples, plot_dir=plot_dir,
                plot_extent=plot_extent, output_path=output_path, use_jac=use_jac),
        x0=[initial_value],
        method='L-BFGS-B',
        jac=True if use_jac else None,
        bounds=[(lower_bound, upper_bound)],
        callback=convergence,
    )
    if convergence.stopped:
        result.success = True
        result.status = 0
        result.message = (f'Parameter {SURFACE_PARAMETER_NAME} changed by at most '
                          f'{PARAMETER_CHANGE_TOL_CM:g} cm for '
                          f'{PARAMETER_STABLE_ITERATIONS} consecutive accepted iterations.')
    print('Saved optimization history to:', output_path)
    update_parameter(float(result.x[0]))
    return result

def collect_parameter_sweep(num_values=100, output_path='parameter_sweep.npy',
                            plot_dir=None):
    """Sample the parameter range; optionally save material slices to plot_dir."""
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
    args = parser.parse_args()
    if args.sweep:
        collect_parameter_sweep(args.num_values, args.output or 'parameter_sweep.npy',
                                plot_dir=args.plot_dir)
        return
    result = optimize_parameter(args.output or 'optimization_history.npy',
                                plot_dir=args.plot_dir, use_jac=not args.no_jac)
    print("Optimization success:", result.success)
    print("Optimized parameter:", result.x[0])
    print("Optimized flux:", result.fun)
    print("Final gradient:", result.jac[0])
    print("Termination message:", result.message)


if __name__ == '__main__':
    main()
