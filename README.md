# OpenMC geometric derivative experiments

This repository explores how changing an iron sphere's radius and position affects neutron flux in a water sphere. It compares OpenMC geometric derivatives with finite differences, uses those derivatives to minimize the water flux, and separates flux contributions by collision count.

The model in `model.py` uses a 14.1 MeV point neutron source at the origin directed uniformly into the positive-x hemisphere. The iron sphere starts at `(4, 5, 0)` cm with radius 2 cm; the water sphere is at `(25, 0, 0)` cm with radius 5 cm. Both sit in a void bounded by a vacuum sphere of radius 1000 cm. Tally 1, `water_flux`, measures collision-estimator flux in the water cell. Geometric derivative 1 differentiates that tally with respect to coefficients of iron surface 1.

## Where to change the experiment

**Edit `config.py` to select the surface parameters used for derivatives.** The scalar sweep and optimization use separate settings:

```python
# Coefficient swept by python optimize.py --sweep:
SURFACE_PARAMETER_NAME = 'y0'

# Ordered coefficients differentiated and optimized together:
OPTIMIZATION_PARAMETER_NAMES = ('r', 'y0', 'x0')
```

Supported iron-sphere coefficients are `r` (radius) and `x0`, `y0`, `z0` (center coordinates), all in cm. For a radius sweep, set `SURFACE_PARAMETER_NAME = 'r'`. To optimize position only, set `OPTIMIZATION_PARAMETER_NAMES = ('x0', 'y0')`; for radius only, use the one-element tuple `('r',)`. Change bounds in `optimize.py`'s `get_constraints()` function.

`model.py`'s `build_model()` assigns these names to `geom_derivative.surface_parameters`. Standalone `python model.py` uses `OPTIMIZATION_PARAMETER_NAMES`; the sweep driver overrides that list with just `SURFACE_PARAMETER_NAME`. Python callers can also pass `parameter_names` explicitly to `build_model()` or `optimize_parameter()`. The differentiated tally and surface are selected by `openmc.GeometricDerivative(flux_tally, iron_sphere, id=1)` in `model.py`. Changing the target surface or tally also requires updating the drivers, which currently reference surface 1, water surface 2, tally 1, and derivative 1. The collision-count collector removes geometric derivatives and always sweeps radius.

**Edit `model.py`'s `build_model()` to adjust the initial model.** Its labeled sections contain the material compositions and densities, sphere centers and radii, outer boundary, source location/direction/energy, transport batches and particles, and flux tally. In particular, the `iron_sphere = openmc.Sphere(...)` constructor sets the optimizer's initial radius and position; `water_sphere = openmc.Sphere(...)` sets the receiving geometry. Use the constructor values as the source of truth, since some nearby comments describe older positions.

Run `python model.py` after those edits to regenerate `model.xml` for the collision-count workflow. `optimize.py` automatically rebuilds and exports the model at the start of each optimization or scalar sweep, so persistent changes to its initial model belong in `model.py`. Revisit parameter bounds when changing the geometry, and regenerate results and matching geometry images for the revised experiment.

## Requirements

- Python with NumPy, SciPy, Matplotlib, and Pillow. The basin-hopping implementation requires a SciPy version supporting the `basinhopping(..., rng=...)` argument.
- An OpenMC build whose Python API and shared library support `GeometricDerivative`, `Model.geometry_derivatives`, and the corresponding `openmc.lib.geometry_derivatives` interface. These experiments rely on those features; a generic OpenMC installation may not provide them. The saved transport report identifies its build as `0.16.1-dev54`, commit `cd62ade62a39fc7220bde11dd18ae7024f5b63e7`.
- Neutron cross-section data for iron, hydrogen, and oxygen, including the `c_H_in_H2O` thermal scattering table. Configure OpenMC's data path, for example with `OPENMC_CROSS_SECTIONS=/path/to/cross_sections.xml`.
- FFmpeg for MP4 animations; GIF output uses Pillow.

Install the supporting Python packages into the environment containing your compatible OpenMC build:

```sh
python -m pip install numpy scipy matplotlib pillow
export OPENMC_CROSS_SECTIONS=/path/to/cross_sections.xml
```

Run commands from the repository root. For plotting on a machine without a display, set `MPLBACKEND=Agg`. Transport defaults to 100 batches of 1,000,000 particles per evaluation, so sweeps and optimization can be expensive. Adjust `model.py` before starting a smaller experiment.

## Main scripts

| File | Purpose and usage |
| --- | --- |
| `model.py` | Builds a fresh model with stable surface, cell, tally, and derivative IDs. `python model.py` exports `model.xml` without running transport. Python callers can use `build_model(parameter_names=...)` or `main(output_path=..., parameter_names=...)`. |
| `optimize.py` | Runs either flux minimization or a scalar derivative sweep. Exports a fresh model, runs transport through the OpenMC shared library, and saves each completed objective evaluation atomically to a structured `.npy` history. See workflows below. |
| `collect_collision_sweep.py` | Loads an existing model XML, removes geometric derivatives, sweeps iron radius, and appends a collision filter to all tallies. Defaults to 100 radius points, collision counts 1–20, score `flux`, seed 10, and `collision_parameter_sweep.npz`. |
| `plot_parameter_sweep.py` | Plots a scalar sweep's flux with ±3 standard deviations, geometric and finite-difference derivatives, their difference, `dJ` and `datt` components, and a component ratio. `--simple` keeps only the flux and two derivative panels. Defaults to `parameter_sweep.npy` and a PNG with the same stem. |
| `plot_collision_sweep.py` | Plots per-collision trends with uncertainty bands and their sum. Defaults to the first tally in `collision_parameter_sweep.npz`; select another with `--tally-id` or `--tally-name`. `--sigma` controls the band width (default 3). |
| `animate_parameter_sweep.py` | Combines saved geometry PNGs with growing scalar sweep plots in evaluation order. Defaults to `parameter_sweep.npy`, `geom_plots/`, and `parameter_sweep_animation.gif`. |
| `animate_optimization.py` | Animates scalar or vector optimization histories, including line-search evaluations and basin jumps. Shows coordinates, flux, derivatives, and flux secants along the traveled path compared with projected gradients. Defaults to `optimization_history.npy`, `optimization_geom/`, and `optimization_history_animation.gif`. |
| `geometry_plots.py` | Shared helper module, without a command-line entrypoint. Computes a fixed XY extent and saves material slices with source markers and run-indexed parameter filenames. Used by the simulation drivers. |
| `config.py` | Shared settings, without a command-line entrypoint. Selects the swept coefficient, optimized coordinates, units, convergence tolerance, flux uncertainty trigger, and basin-hopping defaults. |
| `test_optimization.py` | `unittest` checks for coordinate mapping, vector histories, convergence, and bounded basin hopping. Uses mocked transport and synthetic objectives. |
| `test_optimization_animation.py` | `unittest` checks for path secants, matching geometry filenames, and rendering scalar/vector GIFs using synthetic data. |

Each command-line analysis or simulation driver accepts `--help` for its complete options. `model.py` has no argument parser.

## Workflows

### Sweep and compare derivatives

```sh
python optimize.py --sweep -n 100 -o parameter_sweep.npy --plot-dir geom_plots
python plot_parameter_sweep.py parameter_sweep.npy
python plot_parameter_sweep.py parameter_sweep.npy --simple -o parameter_sweep_simple.png
python animate_parameter_sweep.py parameter_sweep.npy --plot-dir geom_plots
```

The sweep varies `SURFACE_PARAMETER_NAME` from `config.py` (currently `y0`) over the bounds returned by `get_constraints()` (currently −10 to 10 cm for `y0`). It restores the initial coefficient afterward. Derivatives are estimated from the sampled means with `numpy.gradient` for comparison with the transport derivative. Use at least two points for these plots, and use scalar sweep data with unique parameter values.

### Minimize water flux

```sh
python optimize.py --plot-dir optimization_geom
python animate_optimization.py optimization_history.npy --plot-dir optimization_geom
```

The default optimizes `(r, y0, x0)` together with basin hopping around bounded L-BFGS-B local minimizations. Current bounds are 0.01–20 cm for radius and −10–10 cm for `x0` and `y0`. Change `OPTIMIZATION_PARAMETER_NAMES` in `config.py` to select coordinates; the Python function also accepts `parameter_names=('y0', 'x0')`, for example.

```sh
python optimize.py --method local -o local_history.npy --plot-dir local_geom
python optimize.py --no-jac --hops 5 -o numerical_history.npy
python optimize.py --hops 20 --hop-step-fraction 0.5 --hop-temperature 0.01 --hop-seed 10
```

`--no-jac` lets SciPy estimate the optimization gradient numerically, while the simulations still collect geometric derivative results. The local strategy stops after three accepted iterations whose maximum coordinate change is at most `1e-4` cm. Basin hopping uses its configured jump count instead. All evaluations use transport seed 10; the hop seed controls basin proposals separately.

Both optimization and scalar sweeps enable a 1% relative-error trigger on the flux tally, with checks every 10 batches and a maximum of five times the base batch count. Optimization writes the best returned coordinates to `model.xml`. Histories record every objective evaluation, including rejected trials, rather than only accepted iterates.

### Resolve flux by collision count

```sh
python model.py
python collect_collision_sweep.py model.xml -n 100 --min-collision 1 --max-collision 20 --plot-dir collision_geom
python plot_collision_sweep.py collision_parameter_sweep.npz --tally-id 1
```

This workflow always sweeps radius, independently of `SURFACE_PARAMETER_NAME`. It preserves the other coefficients from the supplied XML, so regenerate `model.xml` first to use the initial geometry. Use `--disable-triggers` to remove any uncertainty triggers inherited from that XML; `--score`, `--seed`, and `-o` customize collection. The plotted sum includes only the selected collision bins and is not necessarily the total unfiltered flux. When collapsing other bins, the collector combines standard deviations in quadrature without covariance terms.

### Animation options and tests

Both animation scripts accept `--fps` (default 5), `--dpi` (default 100), and `-o` with a `.gif` or `.mp4` extension:

```sh
python animate_parameter_sweep.py parameter_sweep.npy --plot-dir geom_plots -o parameter_sweep_animation.mp4 --fps 10
MPLBACKEND=Agg python -m unittest test_optimization test_optimization_animation
```

Animations require one matching geometry image for every history row. Use a fresh plot directory for each run: old images with the same run index can cause ambiguous matches. The tests do not run neutron transport, but the optimization tests still import the compatible OpenMC API and shared library.

## Saved data and existing outputs

Current scalar sweep files contain `parameter`, `tally_mean`, `tally_std_dev`, `geometry_derivative`, `dJ`, and `datt`. Optimization histories add `parameter_names`; for multiple coordinates, the parameter and derivative fields are vectors in that saved name order. Load these with `numpy.load(path)`.

Collision `.npz` files contain `radii`, `actual_radii`, `collision_bins`, `tally_ids`, `tally_names`, `mean`, `std_dev`, `collision_sum`, `score_name`, and `model_xml`. The mean and uncertainty arrays have shape `(tallies, radius points, collision bins)`; the sum has shape `(tallies, radius points)`. The existing collision archive has shape `(1, 100, 20)` for its mean array.

The working directory also contains outputs from earlier experiments:

- `parameter_sweep.npy` contains 100 scalar points spanning −7 to 7 cm, unlike the current −10 to 10 cm bounds. Its PNG shows a flux minimum near the central position and broadly similar tallied and finite-difference derivatives. Saved results describe that run, not necessarily the current configuration.
- `optimization_history.npy` and `hops.npy` contain two-coordinate histories; `xy.npy` and `xy_latest.npy` contain three-coordinate histories. Their saved `parameter_names` identify the coordinates. Geometry directories such as `opt_geom/`, `hops_geom/`, and `xy_geom/` and the GIFs preserve earlier visualization runs; verify that a history and image directory match before reusing them.
- `parameter_sweep_old.npy`, `parameter_sweep_gold.npy`, and `optimization_history_deriv.npy` use the older `radius` field instead of `parameter`. Some also lack `dJ` and `datt`. Current plot/animation loaders reject these schemas; regenerate data or explicitly convert it before plotting.
- **`parameter_sweep_n2.py`, `parameter_sweep_n3.py`, and `parameter_sweep_n4.py` are binary NumPy arrays, not Python scripts.** Each holds 100 rows with `radius`, `tally_mean`, `tally_std_dev`, and `geometry_derivative`. Read them with `numpy.load`; their names alone do not establish the experimental settings.
- `model.xml`, split material/geometry/settings/tally XML files, `statepoint.*.h5`, `summary.h5`, and `tallies.out` are OpenMC input exports and simulation outputs. `out.txt` and `out_report.txt` preserve run logs. PNGs, GIFs, MP4s, and geometry image directories are generated visualizations.

The `.gitignore` excludes generated simulation files, numeric results, visualization outputs, caches, and editor backups. It leaves source and test scripts visible. Ignore rules do not remove files already tracked by Git, including the three historical NumPy arrays with `.py` extensions.
