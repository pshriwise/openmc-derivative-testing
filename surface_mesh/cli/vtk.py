"""Command-line adapter for vtk_output."""

import argparse
from pathlib import Path

from ..core.vtk_output import export_run


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runs', nargs='+', type=Path)
    args = parser.parse_args(argv)
    for run in args.runs:
        print(f'{run}: {export_run(run)} VTK files')


if __name__ == '__main__':
    raise SystemExit(main())
