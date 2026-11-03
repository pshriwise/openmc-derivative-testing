"""Optimize a mesh with INT-inspired LargeSteps, Adam, and soft penalties."""

import argparse
from dataclasses import asdict
from pathlib import Path

import numpy as np

from ..mesh.reporting import write_report, file_hash
from ..int_optimizer.config import load_config
from ..int_optimizer.engine import optimize_vertices
from ..int_optimizer.fixtures import sphere, gradient_field, write_obj
from ..int_optimizer.interchange import load_derivatives, load_mesh, write_displaced_h5m


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--int-config', type=Path, required=True,
                        help='JSON containing only the int_inspired namespace')
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--gradient-npz', type=Path)
    source.add_argument('--demo', choices=['localized', 'smooth', 'radial', 'zero'])
    parser.add_argument('--input-h5m', type=Path)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--dry-run', action='store_true', help='Report/OBJ only; do not publish H5M')
    parser.add_argument('--legacy-tally-id', type=int)
    parser.add_argument('--legacy-coordinate-units')
    parser.add_argument('--legacy-derivative-units')
    args = parser.parse_args(argv)
    config = load_config(args.int_config)
    if config.remesh_interval:
        parser.error('Periodic remeshing requires surface_mesh.cli.run to refresh transport derivatives')
    if args.input_h5m and not args.gradient_npz:
        parser.error('--input-h5m requires --gradient-npz')
    snapshot = (load_mesh(args.input_h5m, validate_geometry=config.enable_safeguards)
                if args.input_h5m else None)
    metadata = None
    legacy = (args.legacy_tally_id, args.legacy_coordinate_units, args.legacy_derivative_units)
    if any(v is not None for v in legacy):
        if not all(v is not None for v in legacy):
            parser.error('Provide all three --legacy-* metadata arguments')
        metadata = dict(tally_id=args.legacy_tally_id, objective=config.objective,
                        coordinate_units=args.legacy_coordinate_units,
                        derivative_units=args.legacy_derivative_units)
    if args.demo:
        x, faces = sphere()
        gradient = gradient_field(x, args.demo)
        provenance = dict(source=f'synthetic:{args.demo}', coordinate_units='arbitrary length',
                          derivative_units='synthetic objective / length', objective=config.objective)
    else:
        bundle = load_derivatives(args.gradient_npz, snapshot=snapshot, legacy_metadata=metadata)
        if bundle.metadata['objective'] != config.objective:
            raise ValueError('Derivative metadata objective conflicts with INT config')
        x, faces, gradient = bundle.vertices, bundle.faces, bundle.gradients
        provenance = dict(bundle.metadata, gradient_file=str(args.gradient_npz.resolve()),
                          gradient_file_sha256=file_hash(args.gradient_npz),
                          input_sha256=bundle.input_sha256)
    # Reserve a new run directory before any output; never touch existing runs.
    args.output_dir.mkdir(parents=True, exist_ok=False)
    result = optimize_vertices(x, faces, gradient, config)
    report = dict(method='int_inspired', status=result.status, int_inspired=asdict(config),
                  provenance=provenance, history=result.history, diagnostics=result.diagnostics,
                  dry_run=args.dry_run)
    if snapshot:
        report['selection'] = dict(surface_id=snapshot.surface_id, volume_id=snapshot.volume_id)
    try:
        if snapshot and result.status != 'rejected_step':
            report['h5m_validation'] = write_displaced_h5m(
                snapshot, args.output_dir / 'optimized.h5m', result.vertices, config,
                dry_run=args.dry_run)
    except (ValueError, RuntimeError, ImportError) as exc:
        report.update(status='publication_rejected', publication_error=str(exc))
    write_obj(args.output_dir / 'initial.obj', x, faces)
    write_obj(args.output_dir / 'optimized.obj', result.vertices, faces)
    np.savez_compressed(args.output_dir / 'result.npz', vertices=result.vertices, faces=faces,
                        reference_vertices=x, gradients=gradient,
                        **({'vertex_handles': snapshot.vertex_handles} if snapshot else {}))
    write_report(args.output_dir / 'history.json', report)
    print(f"{report['status']}: {args.output_dir / 'history.json'}")
    return 2 if report['status'] in ('rejected_step', 'publication_rejected') else 0


if __name__ == '__main__':
    raise SystemExit(main())
