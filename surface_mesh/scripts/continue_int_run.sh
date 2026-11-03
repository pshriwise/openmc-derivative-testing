#!/usr/bin/env bash
# Continue an INT run from its last accepted update with the optimizer state it
# had: update numbering (remesh schedule), volume target, learning rate, Adam
# moments, and the seed sequence. Runs saved before Adam states were recorded
# are rebuilt by replaying each accepted update against its saved mesh.
#
# Adaptive sampling is off by default: its plateau test compares three
# consecutive updates against tally noise, which small INT steps never exceed,
# so it raised particles 4x by update 27 of optimization-20261030-053309.
# Transport returns to the run config's particle count.
#
# Usage: ./surface_mesh/scripts/continue_int_run.sh SOURCE_RUN [FINAL_UPDATE]
#        ADAPTIVE_SAMPLING=1 ./surface_mesh/scripts/continue_int_run.sh SOURCE_RUN
# Stop the source run first if it is still going; an update in progress is ignored.
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."

SOURCE=${1:?Usage: continue_int_run.sh SOURCE_RUN [FINAL_UPDATE]}
FINAL_UPDATE=${2:-1250}
OUTPUT="surface_mesh/results/optimization-$(date +%Y%m%d-%H%M%S)"
SAMPLING=(--no-adaptive-sampling)
[[ ${ADAPTIVE_SAMPLING:-0} == 1 ]] && SAMPLING=()

"${PYTHON:-python}" -u -m surface_mesh.cli.run \
  --continue-from "$SOURCE" \
  --iterations "$FINAL_UPDATE" \
  "${SAMPLING[@]}" \
  --output "$OUTPUT"
