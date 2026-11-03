"""Strict derivative bundles and coordinate-only publication via existing I/O."""

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

from ..mesh.constraints import MeshConstraints, load_mesh
from ..mesh.geometry import mesh_arrays
from ..mesh.options import DeformationOptions
from ..mesh.moab_adapter import MoabMesh, staged_mesh, verify_roundtrip
from ..mesh.reporting import file_hash
from .engine import geometry_hash, validate_candidate


@dataclass
class DerivativeBundle:
    vertices: np.ndarray
    faces: np.ndarray
    gradients: np.ndarray
    vertex_ids: np.ndarray
    metadata: dict
    input_sha256: str | None = None


def _ids(values, count):
    ids = np.asarray(values)
    if (ids.shape != (count,) or ids.dtype.kind not in 'iu' or np.any(ids < 0)
            or len(np.unique(ids)) != count):
        raise ValueError('Vertex IDs must be unique nonnegative integers, one per row')
    return ids


def align_gradient_ids(gradient_ids, gradients, vertex_ids):
    """Map explicit derivative IDs to vertex rows, rejecting incomplete maps.

    A geometry-parameter exporter must first explicitly translate its parameter
    IDs to vertex IDs and Cartesian axes; this routine never guesses that map.
    """
    target = _ids(vertex_ids, len(vertex_ids))
    source = _ids(gradient_ids, len(target))
    gradient = np.asarray(gradients, dtype=float)
    if gradient.shape != (len(target), 3) or not np.isfinite(gradient).all():
        raise ValueError('Expected finite gradient XYZ per derivative ID')
    lookup = {int(handle): index for index, handle in enumerate(source)}
    if set(lookup) != set(map(int, target)):
        raise ValueError('Missing or unknown derivative vertex IDs')
    return gradient[[lookup[int(handle)] for handle in target]].copy()


def load_derivatives(path, *, snapshot=None, legacy_metadata=None):
    """Read canonical NPZ or the existing Genesis evaluation sample.npz.

    Canonical metadata is a JSON string with tally_id, coordinate_units,
    derivative_units, objective, and geometry_sha256. H5M bundles additionally
    require vertex_handles and input_sha256. Legacy samples lack objective/units
    metadata, so callers must supply it explicitly through legacy_metadata.
    """
    with np.load(path, allow_pickle=False) as source:
        legacy = 'geometry_derivative' in source
        xkey, gkey = ('coordinates', 'geometry_derivative') if legacy else ('vertices', 'gradients')
        x, f = mesh_arrays(source[xkey], source['faces'])
        x, f = x.copy(), f.copy()
        idkey = 'vertex_handles' if 'vertex_handles' in source else 'vertex_ids'
        ids = _ids(source[idkey], len(x)).copy()
        gradient_ids = source['gradient_vertex_ids'] if 'gradient_vertex_ids' in source else ids
        gradient = align_gradient_ids(gradient_ids, source[gkey], ids)
        if legacy:
            if legacy_metadata is None or snapshot is None:
                raise ValueError('Genesis samples require input H5M and explicit objective/units metadata')
            metadata = dict(legacy_metadata, source_format='genesis_sample',
                            geometry_sha256=geometry_hash(x, f))
            if 'tally_mean' in source:
                metadata['tally_mean'] = float(source['tally_mean'])
        else:
            metadata = json.loads(str(source['metadata']))
        required = ('tally_id', 'coordinate_units', 'derivative_units', 'objective', 'geometry_sha256')
        if (not isinstance(metadata, dict) or any(k not in metadata for k in required)
                or any(not isinstance(metadata[k], str) or not metadata[k].strip()
                       for k in ('coordinate_units', 'derivative_units'))
                or metadata['objective'] not in ('minimize', 'maximize')):
            raise ValueError('Incomplete objective/units/geometry metadata')
        if metadata['geometry_sha256'] != geometry_hash(x, f):
            raise ValueError('Stale derivative reference geometry hash')
        digest = str(source['input_sha256']) if 'input_sha256' in source else None
        if snapshot is not None:
            if idkey != 'vertex_handles' or digest != snapshot.input_sha256:
                raise ValueError('H5M derivatives require matching file hash and MOAB handles')
            lookup = {int(handle): index for index, handle in enumerate(ids)}
            if set(lookup) != set(map(int, snapshot.vertex_handles)):
                raise ValueError('Derivative handles do not match H5M vertices')
            order = np.array([lookup[int(handle)] for handle in snapshot.vertex_handles])
            inverse = np.empty(len(order), dtype=int)
            inverse[order] = np.arange(len(order))
            if (not np.array_equal(x[order], snapshot.coordinates)
                    or not np.array_equal(inverse[f], snapshot.faces)):
                raise ValueError('Derivative coordinates/connectivity differ from current H5M')
            x, f, gradient, ids = x[order], inverse[f], gradient[order], ids[order]
        return DerivativeBundle(x, f, gradient, ids, metadata, digest)


def write_displaced_h5m(snapshot, output_path, vertices, config, *, dry_run=False):
    """Validate and publish unchanged topology/tags using the existing writer.

    Only an isolated single-surface single-volume DAGMC shell is supported;
    load_mesh rejects ambiguous/multiple regions and shared volume boundaries.
    Existing outputs and aliases of the input are refused by the staged writer.
    """
    metrics = validate_candidate(snapshot.coordinates, vertices, snapshot.faces, config)
    if not config.enable_safeguards:
        if dry_run:
            return metrics
        if Path(output_path).suffix.lower() != '.h5m':
            raise ValueError('INT publication requires an .h5m output')
        if file_hash(snapshot.path) != snapshot.input_sha256:
            raise ValueError('Snapshot input changed')
        before = MoabMesh(snapshot.path)
        if (not np.array_equal(before.coordinates, snapshot.coordinates)
                or not np.array_equal(before.faces, snapshot.faces)):
            raise ValueError('Snapshot does not match input geometry')
        with staged_mesh(snapshot.path, output_path) as staged:
            before.set_coordinates(vertices)
            before.write(staged)
            after = MoabMesh(staged)
            verify_roundtrip(before, after, vertices)
            if not np.array_equal(before.vertex_handles, after.vertex_handles):
                raise ValueError('Vertex handles changed during serialization')
            if file_hash(snapshot.path) != snapshot.input_sha256:
                raise ValueError('Snapshot input changed during export')
            metrics['output_sha256'] = file_hash(staged)
        return metrics
    class IntMeshConstraints(MeshConstraints):
        def validate(self, candidate):
            # Run the INT trust/edge checks inside the staging transaction too.
            validate_candidate(snapshot.coordinates, np.asarray(candidate), snapshot.faces, config)
            return super().validate(candidate)

    constraints = IntMeshConstraints(snapshot, options=DeformationOptions(
        preserve_volume=False, check_self_intersections=config.check_self_intersections,
        min_area_ratio=config.min_area_ratio, min_normal_cosine=config.min_normal_cosine))
    if dry_run:
        return constraints.validate(vertices)
    # The existing writer verifies metadata by type-local correspondence. This
    # backend also promises literal handle stability, so check it before the
    # outer staging transaction publishes anything to the user's destination.
    before = MoabMesh(snapshot.path)
    with staged_mesh(snapshot.path, output_path) as staged:
        metrics, _ = constraints.write(vertices, staged)
        after = MoabMesh(staged)
        for kind in range(before.types.MBMAXTYPE):
            if not np.array_equal(before.mb.get_entities_by_type(0, kind),
                                  after.mb.get_entities_by_type(0, kind)):
                raise ValueError('Entity handles changed during H5M serialization')
    return metrics


__all__ = ['DerivativeBundle', 'align_gradient_ids', 'load_derivatives',
           'load_mesh', 'write_displaced_h5m']
