"""Export the sphere transport model."""

import argparse
from pathlib import Path
from ..core.model import build_model


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('model.xml'))
    args = parser.parse_args(argv)
    build_model().export_to_model_xml(args.output)


if __name__ == '__main__':
    raise SystemExit(main())
