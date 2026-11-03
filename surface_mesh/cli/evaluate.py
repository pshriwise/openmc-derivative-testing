"""Command-line adapter for evaluate."""

import argparse
import json
from pathlib import Path

from ..core import evaluate
from ..core.settings import IntRunSettings


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mesh', type=Path, required=True)
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--settings', type=Path, required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--budget', type=int, default=1)
    parser.add_argument('--batch-scores', action='store_true')
    parser.add_argument('--settings-kind', choices=['int'], default='int')
    args = parser.parse_args(argv)
    evaluate.worker(args.mesh, args.directory, IntRunSettings(**json.loads(args.settings.read_text())),
           args.seed, args.budget, batch_scores=args.batch_scores, progress=args.settings_kind == 'int')


if __name__ == '__main__':
    raise SystemExit(main())
