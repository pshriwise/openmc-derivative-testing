#!/usr/bin/env bash
# Mimic the INT repository's optimizer (https://github.com/joyDeng/INT,
# optimization.py) using every other setting from a saved surface_mesh run.
#
# Kept from INT: one Adam step per transport gradient with the optimizer kept
# between updates, the remesh schedule, and the post-remesh learning rate per
# edge length. INT's values were chosen for a scene about two units across, so
# length-dependent settings are converted with LENGTH_SCALE (cm per INT unit):
#   learning_rate x L       range_weight / L^2
# Two settings deliberately differ from INT for this mesh:
#   LAMBDA=5 (INT 35): LargeSteps uses a combinatorial Laplacian, so its
#     smoothing depends on mesh resolution. At 35 on the 516-vertex 2 cm sphere
#     only translation survives; 5 lets shape changes through.
#   VOLUME_WEIGHT=1e-3 (INT 5, i.e. 5e-6 at L = 10 cm): the scaled weight lets the
#     volume grow without limit because added iron lowers the flux; 1e-3 balances
#     the transport gradient at about 1% volume change.
#
# Overrides: LAMBDA=35 VOLUME_WEIGHT=5e-6 ./surface_mesh/scripts/run_int_repo_schedule.sh SOURCE_RUN
#            LENGTH_SCALE=8 ./surface_mesh/scripts/run_int_repo_schedule.sh SOURCE_RUN
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."

SOURCE=${1:?Usage: run_int_repo_schedule.sh SOURCE_RUN}
OUTPUT="surface_mesh/results/optimization-$(date +%Y%m%d-%H%M%S)"
LENGTH_SCALE=${LENGTH_SCALE:-10}
LAMBDA=${LAMBDA:-5}
VOLUME_WEIGHT=${VOLUME_WEIGHT:-1e-3}
CONFIG=$(mktemp --suffix=.json)
trap 'rm -f -- "$CONFIG"' EXIT

# Start from the source run's INT config. Keep the reference schedule here so
# this launcher does not depend on a separate preset file. The values match the
# former int_inspired_int_repo.json; bounds and other run-specific options are
# inherited from the source. The launcher saves the merged config in its output.
"${PYTHON:-python}" - "$SOURCE/int_inspired_config.json" \
  "$CONFIG" "$LENGTH_SCALE" "$LAMBDA" "$VOLUME_WEIGHT" <<'EOF'
import json
from pathlib import Path
import sys

run, output, scale, large_steps_lambda, volume_weight = sys.argv[1:]
scale = float(scale)
config = json.loads(Path(run).read_text())['int_inspired']
config.update({
    'large_steps_lambda': 35.0,
    'learning_rate': 0.005,
    'iterations': 1,
    'volume_weight': 5.0,
    'range_weight': 2.0,
    'persist_adam_state': True,
    'remesh_interval': 100,
    'remesh_iterations': 5,
    'remesh_target_edge_length': None,
    'remesh_adam_state': 'reset',
    'remesh_learning_rate_per_edge': 0.07,
    'remesh_halve_edge_rounds': [3, 5],
})
config['learning_rate'] *= scale
config['range_weight'] /= scale**2
config['large_steps_lambda'] = float(large_steps_lambda)
config['volume_weight'] = float(volume_weight)
Path(output).write_text(json.dumps({'int_inspired': config}, indent=2) + '\n')
print(f"INT schedule at {scale:g} cm per unit: learning_rate {config['learning_rate']:g}, "
      f"range_weight {config['range_weight']:g}; large_steps_lambda {config['large_steps_lambda']:g}, "
      f"volume_weight {config['volume_weight']:g}", flush=True)
EOF

# INT runs 1250 optimizer iterations; each is one fresh transport evaluation here.
"${PYTHON:-python}" -u -m surface_mesh.cli.run \
  --input "$SOURCE/initial.h5m" \
  --int-config "$CONFIG" \
  --run-config "$SOURCE/run_settings.json" \
  --iterations 1250 \
  --output "$OUTPUT"
