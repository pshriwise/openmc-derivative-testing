#!/usr/bin/env bash
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."

# Reuse the saved input and settings; the output directory must be new.
SOURCE=${1:?Usage: reproduce_int_remesh_long.sh SOURCE_RUN}
"${PYTHON:-python}" -u -m surface_mesh.cli.run \
  --input "$SOURCE/initial.h5m" \
  --int-config "$SOURCE/int_inspired_config.json" \
  --run-config "$SOURCE/run_settings.json" \
  --output "surface_mesh/results/optimization-$(date +%Y%m%d-%H%M%S)"
