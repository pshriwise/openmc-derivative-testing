# Saved run configurations

Three distinct configuration pairs extracted unchanged from recent runs.
Each optimizer file uses the `int_inspired` namespace; each run file uses
`int_run`. The original saved run files remain unchanged.

| Profile | Optimizer | Run settings | Source run |
| --- | --- | --- | --- |
| persistent_adam | [persistent_adam_optimizer.json](persistent_adam_optimizer.json) | [persistent_adam_run.json](persistent_adam_run.json) | `optimization-20261030-053309` |
| remesh_every_update | [remesh_every_update_optimizer.json](remesh_every_update_optimizer.json) | [remesh_every_update_run.json](remesh_every_update_run.json) | `optimization-20261029-212927` |
| int_scaled_reference | [int_scaled_reference_optimizer.json](int_scaled_reference_optimizer.json) | [int_scaled_reference_run.json](int_scaled_reference_run.json) | `optimization-20261029-201200` |

Run from the repository root, supplying the mesh used for your experiment:

```bash
python -m surface_mesh --input path/to/initial.h5m \
  --int-config surface_mesh/configs/persistent_adam_optimizer.json \
  --run-config surface_mesh/configs/persistent_adam_run.json \
  --output surface_mesh/results/new_run
```

These snapshots retain their original iteration counts, sampling budgets,
adaptive-sampling settings and safeguard settings. CLI overrides remain available.

## Fresh runs

Start the tweaked INT schedule for 500 outer updates with an explicit initial mesh:

```bash
./surface_mesh/scripts/run_int_fresh.sh path/to/initial.h5m
```

This defaults to `persistent_adam`: one Adam step per transport evaluation,
lambda 5, learning rate 0.05, volume weight 0.001, range weight 0.02, and remeshing
every 100 updates. Adaptive sampling follows `persistent_adam_run.json`.
Use `ITERATIONS=100` to shorten the run or supply an output directory as the
second argument. The script prefers the repository `.venv`; `PYTHON` overrides it.

The earlier 500-update experiment (`optimization-20261029-212927`) used a different
profile, with 300 inner Adam steps and remeshing after every outer update:

```bash
PROFILE=remesh_every_update ./surface_mesh/scripts/run_int_fresh.sh path/to/initial.h5m
```

Both launch fresh optimization with new Adam state, without continuation from a
saved run. The presets preserve their existing optional safeguard settings.
