"""Command-line adapter for backfill_solid."""

import argparse
from pathlib import Path

from ..core.backfill_solid import backfill


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runs', type=Path, nargs='+')
    args = parser.parse_args(argv)
    for root in args.runs:
        print(f'{root}: {backfill(root)} distinct meshes', flush=True)


if __name__ == '__main__':
    raise SystemExit(main())
