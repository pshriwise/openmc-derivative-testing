#!/usr/bin/env bash
# Fresh OpenMC/INT optimization using the curated presets under configs/.
# Default: 500 outer updates, one Adam step per fresh transport gradient,
# persistent Adam, lambda=5, learning_rate=0.05, volume_weight=0.001,
# range_weight=0.02, remesh every 100 updates, reset Adam after remeshing,
# and learning-rate/edge refinement on the saved INT schedule.
# Sampling/rendering settings (including adaptive sampling) come from the run preset.
#
# Usage: ./surface_mesh/scripts/run_int_fresh.sh INPUT_MESH [OUTPUT_DIR]
# ITERATIONS=100 ./surface_mesh/scripts/run_int_fresh.sh INPUT_MESH
# PROFILE=remesh_every_update ./surface_mesh/scripts/run_int_fresh.sh INPUT_MESH
# The latter selects the 500-step experiment's 300 inner steps/update and remesh
# every update, without persistent Adam or adaptive particle sampling.
# Paths are relative to the repository root unless absolute.
# PYTHON overrides the repository .venv interpreter when one is available.
set -euo pipefail

if [[ ${1:-} == --help || ${1:-} == -h ]]; then
  echo "Usage: $0 INPUT_MESH [OUTPUT_DIR]"
  echo "Options: ITERATIONS=500 PROFILE=persistent_adam PYTHON=<interpreter>"
  echo "Profiles: persistent_adam, remesh_every_update"
  exit 0
fi
if (( $# < 1 || $# > 2 )); then
  echo "Usage: $0 INPUT_MESH [OUTPUT_DIR]" >&2
  exit 2
fi

cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."

PYTHON_COMMAND=${PYTHON:-python}
if [[ -z ${PYTHON:-} && -x .venv/bin/python ]]; then
  PYTHON_COMMAND="$PWD/.venv/bin/python"
fi

PROFILE=${PROFILE:-persistent_adam}
ITERATIONS=${ITERATIONS:-500}
case "$PROFILE" in
  persistent_adam|remesh_every_update) ;;
  *) echo "Unknown profile: $PROFILE" >&2; exit 2 ;;
esac
if [[ ! "$ITERATIONS" =~ ^[1-9][0-9]*$ ]]; then
  echo "ITERATIONS must be a positive integer" >&2
  exit 2
fi

INPUT=$1
OUTPUT=${2:-surface_mesh/results/int_fresh_${PROFILE}-$(date +%Y%m%d-%H%M%S)}
OPTIMIZER_CONFIG="surface_mesh/configs/${PROFILE}_optimizer.json"
RUN_CONFIG="surface_mesh/configs/${PROFILE}_run.json"
for file in "$INPUT" "$OPTIMIZER_CONFIG" "$RUN_CONFIG"; do
  if [[ ! -f "$file" ]]; then
    echo "Missing input file: $file" >&2
    exit 1
  fi
done
if [[ -e "$OUTPUT" || -L "$OUTPUT" ]]; then
  echo "Output must be a new directory: $OUTPUT" >&2
  exit 1
fi

exec "$PYTHON_COMMAND" -u -m surface_mesh.cli.run \
  --input "$INPUT" \
  --int-config "$OPTIMIZER_CONFIG" \
  --run-config "$RUN_CONFIG" \
  --iterations "$ITERATIONS" \
  --output "$OUTPUT"
