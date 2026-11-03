"""Command-line adapter for prepare_initial_mesh."""

import argparse
import json
from pathlib import Path

from ..core.preparation import prepare, INPUT_MESH, OUTPUT_MESH, TARGET_RADIUS_CM


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=INPUT_MESH)
    parser.add_argument('--output', type=Path, default=OUTPUT_MESH)
    parser.add_argument('--radius', type=float, default=TARGET_RADIUS_CM)
    args = parser.parse_args(argv)
    print(json.dumps(prepare(args.input, args.output, args.radius), indent=2))


if __name__ == '__main__':
    raise SystemExit(main())
