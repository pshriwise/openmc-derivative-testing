"""Command-line adapter for mesh_motion."""

import argparse
import json
from pathlib import Path

from ..core.mesh_motion import render_motion


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    args = parser.parse_args(argv)
    print(json.dumps(render_motion(args.run), indent=2))


if __name__ == '__main__':
    raise SystemExit(main())
