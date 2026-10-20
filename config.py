"""Shared settings for surface-parameter simulations and visualizations."""

SURFACE_PARAMETER_NAME = 'r'
# Ordered coordinates optimized together, e.g. ('x0', 'y0').
OPTIMIZATION_PARAMETER_NAMES = ['r', 'x0', 'y0']
SURFACE_PARAMETER_UNITS = 'cm'

# Run all cases with: python optimize.py --sweep --all-cases
# Unspecified iron coefficients and tally options use model.py defaults.
# iron defines the initial geometry; sweep values replace the swept coefficient.
# Omit bounds (or set them to None) to use optimize.py's get_constraints().
SWEEP_CASES = {
    'radius_centered': {
        'parameter': 'r',
        'num_values': 100,
        'iron': {'r': 2.0, 'x0': 0.0, 'y0': 0.0, 'z0': 0.0},
        'tally': {'score': 'flux'},
    },
    'radius_offset': {
        'parameter': 'r',
        'num_values': 100,
        'iron': {'r': 2.0, 'x0': 4.0, 'y0': 0.0, 'z0': 0.0},
        'tally': {'score': 'flux'},
    },
    'y_sweep': {
        'parameter': 'y0',
        'num_values': 100,
        'iron': {'r': 3.0, 'x0': 6.0, 'y0': 0.0, 'z0': 0.0},
        'tally': {'score': 'flux'},
    },
    'radius_centered_heating': {
        'parameter': 'r',
        'num_values': 100,
        'iron': {'r': 2.0, 'x0': 0.0, 'y0': 0.0, 'z0': 0.0},
        'tally': {'score': 'heating-local'},
    },
    'radius_offset_heating': {
        'parameter': 'r',
        'num_values': 100,
        'iron': {'r': 2.0, 'x0': 4.0, 'y0': 0.0, 'z0': 0.0},
        'tally': {'score': 'heating-local'},
    },
    'y_sweep_heating': {
        'parameter': 'y0',
        'num_values': 100,
        'iron': {'r': 3.0, 'x0': 6.0, 'y0': 0.0, 'z0': 0.0},
        'tally': {'score': 'heating-local'},
    },
    'radius_centered_absorption': {
        'parameter': 'r',
        'num_values': 100,
        'iron': {'r': 2.0, 'x0': 0.0, 'y0': 0.0, 'z0': 0.0},
        'tally': {'score': 'absorption'},
    },
    'radius_offset_absorption': {
        'parameter': 'r',
        'num_values': 100,
        'iron': {'r': 2.0, 'x0': 4.0, 'y0': 0.0, 'z0': 0.0},
        'tally': {'score': 'absorption'},
    },
    'y_sweep_absorption': {
        'parameter': 'y0',
        'num_values': 100,
        'iron': {'r': 3.0, 'x0': 6.0, 'y0': 0.0, 'z0': 0.0},
        'tally': {'score': 'absorption'},
    },
}

PARAMETER_CHANGE_TOL_CM = 1e-4
PARAMETER_STABLE_ITERATIONS = 3

FLUX_TRIGGER_REL_ERR = 1e-2
TRIGGER_MAX_BATCHES_FACTOR = 5
TRIGGER_BATCH_INTERVAL = 10

# Basin-hopping wraps bounded L-BFGS-B local minimizations.
OPTIMIZATION_METHOD = 'basinhopping'
BASIN_HOP_ITERATIONS = 20
BASIN_HOP_STEP_FRACTION = 0.5  # Fraction of each coordinate's bound width.
BASIN_HOP_TEMPERATURE = 0.01  # In flux-objective units.
BASIN_HOP_SEED = 10
