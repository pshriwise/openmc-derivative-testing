"""Command-line adapter for model_box."""

import argparse
from pathlib import Path

from ..core.model_box import build_model, INPUT_MESH


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mesh', type=Path, default=INPUT_MESH)
    parser.add_argument('--output', type=Path, default=Path('model_box.xml'))
    args = parser.parse_args(argv)
    build_model(args.mesh).export_to_model_xml(args.output)


if __name__ == '__main__':
    raise SystemExit(main())
