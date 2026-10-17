"""Shared settings for surface-parameter simulations and visualizations."""

SURFACE_PARAMETER_NAME = 'r'
# Ordered coordinates optimized together, e.g. ('x0', 'y0').
OPTIMIZATION_PARAMETER_NAMES = ['r']
SURFACE_PARAMETER_UNITS = 'cm'

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
