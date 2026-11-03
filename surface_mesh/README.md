# OpenMC / INT surface-mesh optimization

This workflow runs OpenMC transport, obtains vertex-coordinate derivatives, and
updates a single closed DAGMC shell with Mitsuba LargeSteps and Adam. OpenMC owns
the physics; `surface_mesh.int_optimizer` owns the INT update and
`surface_mesh.mesh` owns H5M structure queries, validation, and safe I/O. INT itself is a reference, not a runtime dependency.

## Setup

Use Python 3.11+ and an OpenMC build with the experimental XDG and geometric
vertex-derivative APIs. A stock OpenMC release is insufficient. The Python API
and shared library must come from compatible builds. Compiled PyMOAB bindings
must be available in the environment; pip does not install them here.

From this repository root:

```bash
python -m pip install -r surface_mesh/requirements.txt
export OPENMC_CROSS_SECTIONS=/path/to/nndc_hdf5/cross_sections.xml
```

The tested numerical versions are in `surface_mesh/requirements-tested.txt`.
No separate deformation repository is needed. Rendering also needs
OpenMC's native solid renderer and a working VTK backend for mesh-motion plots.

## First run

The included `sphere_coarse.h5m` is a small starting mesh. Run from this repository's
root, using a new output directory:

```bash
python -m surface_mesh --input surface_mesh/sphere_coarse.h5m --model box \
  --int-config surface_mesh/int_inspired_config.json \
  --run-config surface_mesh/int_run_config.json \
  --particles 10000 --batches 10 --iterations 2 \
  --output surface_mesh/results/first_run
```

This is a smoke-run budget, not a convergence study. `surface_mesh.cli.run` remains
an equivalent entry point. Every completed outer update gets fresh transport and
derivatives on its published mesh.

## Configuration and retained capabilities

The three distinct recent run profiles are preserved in [configs/](configs/README.md),
each with its optimizer and transport settings.

| File / tool | Purpose |
| --- | --- |
| `int_inspired_config.json` (`int_inspired`) | LargeSteps, Adam, volume/range penalties, optional safeguards |
| `int_inspired_remesh.json` | Projected Botsch remeshing preset |
| `int_inspired_int_repo.json` | Reference INT optimizer schedule in INT units; see unit conversion in the guide |
| `int_run_config.json` (`int_run`) | Transport, model, sampling, camera, image and VTK controls |
| `int_run_adaptive.json` | Adaptive particle-sampling preset |
| `run_int --continue-from RUN` | Resume numbering, volume target, Adam state, remesh schedule and seed sequence |
| `int_deform` | Apply a saved derivative without running transport |
| `compare_int` | Compare penalized/unpenalized INT against a Cartesian diagnostic reference |
| `prepare_initial_mesh` | Prepare a different spherical starting radius |

Persistent Adam moments, remesh state reset/transfer, edge-based learning-rate
scheduling, adaptive sampling, progress logs, provenance, and derivative mapping
checks remain available. The optimizer and run JSON namespaces are independent;
unknown settings fail rather than silently selecting another algorithm.

Runs retain slice and solid images, objective CSV/PNG/SVG with uncertainty,
combined and solid GIFs, raw derivative and motion VTK files, mesh-motion plots,
inner optimization diagnostics, and optimizer checkpoints. Regenerate artifacts:

```bash
python -m surface_mesh.cli.report surface_mesh/results/first_run
python -m surface_mesh.cli.vtk surface_mesh/results/first_run
python -m surface_mesh.cli.backfill surface_mesh/results/first_run
python -m surface_mesh.cli.mesh_motion surface_mesh/results/first_run
```

See [the detailed INT guide](INT_INSPIRED.md) for remeshing, continuation,
sampling policy, units, and all saved artifacts. Soft penalties approximate volume
and range constraints. Acceptance records a completed update and transport;
observed tally improvement is diagnostic, not a statistical acceptance test.
The presets retain their current safeguard settings; enable `enable_safeguards`
for the optional geometry/collision/container/source checks.

## Tests and migration

```bash
OMP_NUM_THREADS=2 python -m pytest -q surface_mesh/tests
```

The numerical engine tests can run without OpenMC; native I/O tests require
PyMOAB, and coupled tests need the experimental OpenMC API even when transport
is replaced by an analytical evaluator.

The home-rolled `StepSolver`, quadratic direction solve, projection/restoration,
backtracking, old optimization/sweep launchers and their `config.py` settings have
been removed. Use `surface_mesh` / `surface_mesh.cli.run` with the two INT JSON configurations.
The top-level analytic-sphere experiments elsewhere in this repository are separate.
Saved result directories are unchanged. Superseded experiment plans and results
are retained under `docs/archive/` as historical records; their commands describe
the old implementation. Solid backfill now requires an INT run.

## Code organization

| Directory | Responsibility |
| --- | --- |
| `core/` | Reusable run loop, transport worker, continuation, sampling, model builders, history, plotting, exports and mesh preparation |
| `cli/` | Parse arguments, load configurations and call the reusable services |
| `int_optimizer/` | LargeSteps/Adam, penalties, derivative interchange, remeshing and optimizer state |
| `mesh/` | Ordered H5M snapshots, structure queries, staged writes and geometry/collision validation |
| `tests/` | All surface-mesh tests |

Use `python -m surface_mesh` (or `python -m surface_mesh.cli.run`) to optimize.
Other commands live directly under `cli/`, for example:

```bash
python -m surface_mesh.cli.report surface_mesh/results/my_run
python -m surface_mesh.cli.vtk surface_mesh/results/my_run
python -m surface_mesh.cli.deform --help
```

The redundant top-level compatibility command files and optimizer CLI aliases
have been removed. Python callers import reusable functions from `core/`,
`mesh/`, or `int_optimizer/`.

Python callers use the core functions directly with explicit settings:

```python
from surface_mesh.core.run import run
from surface_mesh.core.settings import IntRunSettings
from surface_mesh.int_optimizer import IntConfig

history = run(
    input_mesh="surface_mesh/sphere_coarse.h5m",
    output="surface_mesh/results/python_run",
    int_config=IntConfig(iterations=1),
    settings=IntRunSettings(model_variant="box", max_iterations=2),
)
```

The core run loop takes settings objects and never parses command-line arguments
or selects a preset JSON file. `cli/run.py` owns preset selection and overrides;
`core/settings.py` defines and validates run settings. Continuation reads saved
run settings and checkpoints to reconstruct the prior state. Edit model physics
in `core/model.py` or `core/model_box.py`; `cli/model.py` and `cli/model_box.py` launch XML export.
Moving the model sources changes their recorded hashes. Continuing a run from
before this reorganization may require `--allow-changed-model` after checking the
model changes.

The optional Cubit generator lives in `core/sphere.py` with its adapter in
`cli/make_sphere.py`.

The required mesh/INT code was extracted from `surface-mesh-deformation`. Its
packaging, legacy deformation algorithms, converter and historical examples are
not dependencies. Mesh inspection and coordinate updates use PyMOAB directly;
PyDAGMC is optional for the adapter's convenience model view. See
[optimizer details](OPTIMIZER.md) for the numerical API.
