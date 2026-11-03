"""Launch OpenMC/INT optimization from JSON settings and CLI overrides."""

import argparse
from dataclasses import replace
from pathlib import Path

from ..core.run import DEFAULT_INPUT, run
from ..core.settings import load_run_config
from ..int_optimizer.config import load_config

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--int-config', type=Path)
    parser.add_argument('--run-config', type=Path)
    parser.add_argument('--continue-from', type=Path, metavar='RUN',
                        help='Resume an INT run from its last accepted update; configs default to its own')
    parser.add_argument('--allow-changed-model', action='store_true',
                        help='Continue even if the model source or OpenMC library changed')
    parser.add_argument('--no-adaptive-sampling', action='store_true',
                        help='Disable adaptive sampling; a continuation returns to --particles or the run config')
    parser.add_argument('--model', choices=['sphere', 'box'])
    parser.add_argument('--iterations', type=int,
                        help='Final outer update number, each with a fresh transport evaluation')
    parser.add_argument('--inner-iterations', type=int, help='Adam steps per frozen-gradient solve')
    for name in ('particles', 'batches', 'threads', 'seed'):
        parser.add_argument(f'--{name}', type=int)
    args = parser.parse_args(argv)
    here = Path(__file__).resolve().parents[1]
    defaults = (here / 'int_inspired_config.json', here / 'int_run_config.json')
    if args.continue_from is not None:
        defaults = (args.continue_from / 'int_inspired_config.json', args.continue_from / 'run_settings.json')
    updates = {key: getattr(args, arg) for key, arg in
               [('model_variant', 'model'), ('max_iterations', 'iterations'),
                ('particles', 'particles'), ('batches', 'batches'), ('threads', 'threads'), ('seed', 'seed')]
               if getattr(args, arg) is not None}
    cfg = replace(load_run_config(args.run_config or defaults[1]), **updates)
    if args.no_adaptive_sampling:
        cfg = replace(cfg, adaptive_sampling=replace(cfg.adaptive_sampling, enabled=False))
    optimizer = load_config(args.int_config or defaults[0])
    if args.inner_iterations is not None:
        optimizer = replace(optimizer, iterations=args.inner_iterations)
    continuation = None
    if args.continue_from is not None:
        from ..core.continuation import continuation_state
        continuation = continuation_state(args.continue_from, optimizer, cfg)
    input_mesh = args.input or (continuation.mesh if continuation else DEFAULT_INPUT)
    result = run(input_mesh, args.output, optimizer, cfg, continuation=continuation,
                 allow_changed_model=args.allow_changed_model)
    print(f"{result['status']}: {args.output / result['accepted_mesh']}", flush=True)
    return 2 if result['status'] == 'geometry_rejected' else 0


if __name__ == '__main__':
    raise SystemExit(main())
