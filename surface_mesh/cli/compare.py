"""Compare INT penalty settings and a Cartesian reference, without transport."""

import argparse
from pathlib import Path


from surface_mesh.int_optimizer.comparison import DirectConfig, compare_methods
from surface_mesh.int_optimizer.config import load_config
from surface_mesh.int_optimizer.fixtures import sphere, gradient_field
from surface_mesh.int_optimizer.interchange import load_mesh, load_derivatives


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--int-config', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--case', choices=['all', 'localized', 'smooth', 'radial', 'zero'], default='all')
    parser.add_argument('--radius', type=float, default=1.)
    parser.add_argument('--subdivisions', type=int, default=2)
    parser.add_argument('--input-h5m', type=Path)
    parser.add_argument('--gradient-npz', type=Path)
    parser.add_argument('--derivative-units', help='Required for existing Genesis samples')
    parser.add_argument('--coordinate-units', default='cm')
    parser.add_argument('--tally-id', type=int, default=1)
    parser.add_argument('--direct-learning-rate', type=float, default=.005)
    parser.add_argument('--direct-iterations', type=int, default=20)
    args = parser.parse_args(argv)
    config = load_config(args.int_config)
    direct = DirectConfig(args.direct_learning_rate, args.direct_iterations)
    if bool(args.input_h5m) != bool(args.gradient_npz):
        parser.error('Provide --input-h5m and --gradient-npz together')
    if args.input_h5m:
        snapshot = load_mesh(args.input_h5m)
        metadata = (dict(tally_id=args.tally_id, coordinate_units=args.coordinate_units,
                         derivative_units=args.derivative_units, objective=config.objective)
                    if args.derivative_units else None)
        bundle = load_derivatives(args.gradient_npz, snapshot=snapshot, legacy_metadata=metadata)
        if bundle.metadata['objective'] != config.objective:
            raise ValueError('Objective sign differs between derivative and INT config')
        x, faces = bundle.vertices, bundle.faces
        cases = {'openmc': bundle.gradients}
        provenance = dict(bundle.metadata, input_sha256=snapshot.input_sha256)
    else:
        x, faces = sphere(args.subdivisions, args.radius)
        names = ['localized', 'smooth', 'radial', 'zero'] if args.case == 'all' else [args.case]
        cases = {name: gradient_field(x, name) for name in names}
        provenance = dict(source='synthetic', radius=args.radius, subdivisions=args.subdivisions)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    for name, gradient in cases.items():
        compare_methods(x, faces, gradient, config, args.output_dir / name,
                        direct_config=direct,
                        provenance=dict(provenance, case=name))
    print(args.output_dir)


if __name__ == '__main__':
    raise SystemExit(main())
