"""Command-line adapter for report."""

import argparse
from pathlib import Path

from ..core.report import regenerate


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runs', nargs='+', type=Path)
    args = parser.parse_args(argv)
    for root in args.runs:
        regenerate(root)
        print(f'Updated reports: {root}', flush=True)


if __name__ == '__main__':
    raise SystemExit(main())
