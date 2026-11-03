#!/usr/bin/env bash
# Start a fresh run from results/int_remesh_long_reproduction's saved mesh and
# exact JSON settings, using the current framework and OpenMC installation.
# Defaults: 100 outer updates, 300 Adam steps/update, remesh every update,
# 50,000 particles/batch, 50 batches, two threads, no adaptive sampling.
#
# Usage: ./surface_mesh/scripts/reproduce_int_remesh_long_reproduction.sh [OUTPUT_DIR]
# Prefer the repository .venv when available. Set PYTHON to override it.
# Relative output paths are interpreted from the repository root.
set -euo pipefail

if [[ ${1:-} == --help || ${1:-} == -h ]]; then
  echo "Usage: $0 [OUTPUT_DIR]"
  exit 0
fi
if (( $# > 1 )); then
  echo "Usage: $0 [OUTPUT_DIR]" >&2
  exit 2
fi

cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."

PYTHON_COMMAND=${PYTHON:-python}
if [[ -z ${PYTHON:-} && -x .venv/bin/python ]]; then
  PYTHON_COMMAND="$PWD/.venv/bin/python"
fi

SOURCE=surface_mesh/results/int_remesh_long_reproduction
OUTPUT=${1:-surface_mesh/results/int_remesh_long_reproduction-$(date +%Y%m%d-%H%M%S)}

for file in initial.h5m int_inspired_config.json run_settings.json; do
  if [[ ! -f "$SOURCE/$file" ]]; then
    echo "Missing saved run file: $SOURCE/$file" >&2
    exit 1
  fi
done
if [[ -e "$OUTPUT" || -L "$OUTPUT" ]]; then
  echo "Output must be a new directory: $OUTPUT" >&2
  exit 1
fi

exec "$PYTHON_COMMAND" -u -m surface_mesh.cli.run \
  --input "$SOURCE/initial.h5m" \
  --int-config "$SOURCE/int_inspired_config.json" \
  --run-config "$SOURCE/run_settings.json" \
  --output "$OUTPUT"
