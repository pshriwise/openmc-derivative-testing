"""Geometry services for external constrained optimizers.

The caller chooses final coordinates. This API never projects, scales, smooths,
or repairs them; rejected candidates leave the input and destination untouched.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .collision import CollisionGuard, locally_injective_path
from .geometry import signed_volume, volume_centroid, quality_metrics
from .moab_adapter import MoabMesh, staged_mesh, verify_roundtrip
from .options import DeformationOptions
from .validation import check_candidate, volume_tolerance
from .reporting import file_hash
from .topology import validate_mesh


@dataclass
class MeshSnapshot:
    path: Path
    input_sha256: str
    coordinates: np.ndarray
    faces: np.ndarray
    vertex_handles: np.ndarray
    surface_id: int
    volume_id: int


def load_mesh(path, *, validate_geometry=True):
    """Load a single-volume DAGMC mesh without editing its metadata.

    Geometry validation is enabled by default. INT can
    disable it; file structure and metadata checks still apply.
    """
    path = Path(path).resolve()
    if path.suffix.lower() != '.h5m':
        raise ValueError('Constraint API requires an existing DAGMC .h5m file')
    digest = file_hash(path)
    mesh = MoabMesh(path)
    if validate_geometry:
        validate_mesh(mesh.coordinates, mesh.faces)
    mesh.reject_external_payloads()
    mesh.reject_cached_geometry()
    ids = mesh.inspect_dagmc()
    if file_hash(path) != digest:
        raise ValueError('Input changed during loading')
    return MeshSnapshot(path, digest, mesh.coordinates.copy(), mesh.faces.copy(),
                        mesh.vertex_handles.copy(), **ids)


class MeshConstraints:
    """Evaluate fixed volume/centroid targets and collision-free updates.

    Targets should be taken from the initial mesh and retained for the entire
    optimization. Centroid tolerance is an absolute distance in mesh units.
    An optional vertex exclusion sphere uses mesh-local coordinates and mesh
    length units. Vertices and their linear update paths must stay at least
    ``vertex_exclusion_radius`` from its center. This does not constrain face
    interiors. A zero radius disables the exclusion.
    """

    def __init__(self, snapshot, *, options=None, volume_target=None,
                 centroid_target=None, centroid_atol=1e-8,
                 vertex_exclusion_center=None, vertex_exclusion_radius=0.0):
        self.snapshot = snapshot
        self.options = options or DeformationOptions()
        if volume_target is not None and (not np.isfinite(volume_target) or volume_target <= 0):
            raise ValueError('Volume target must be positive and finite')
        if self.options.preserve_volume and volume_target is None:
            volume_target = signed_volume(snapshot.coordinates, snapshot.faces)
        self.volume_target = volume_target if self.options.preserve_volume else None
        self.centroid_target = None if centroid_target is None else np.asarray(centroid_target, dtype=float)
        if self.centroid_target is not None and (self.centroid_target.shape != (3,)
                                                 or not np.isfinite(self.centroid_target).all()):
            raise ValueError('Centroid target must be a finite three-vector')
        if not np.isfinite(centroid_atol) or centroid_atol <= 0:
            raise ValueError('Centroid tolerance must be positive and finite')
        self.centroid_atol = centroid_atol
        if not np.isfinite(vertex_exclusion_radius) or vertex_exclusion_radius < 0:
            raise ValueError('Vertex exclusion radius must be finite and nonnegative')
        self.vertex_exclusion_radius = float(vertex_exclusion_radius)
        self.vertex_exclusion_center = (None if vertex_exclusion_center is None
                                        else np.array(vertex_exclusion_center, dtype=float, copy=True))
        if self.vertex_exclusion_center is not None and (
                self.vertex_exclusion_center.shape != (3,)
                or not np.isfinite(self.vertex_exclusion_center).all()):
            raise ValueError('Vertex exclusion center must be a finite three-vector')
        if self.vertex_exclusion_radius > 0 and self.vertex_exclusion_center is None:
            raise ValueError('A positive vertex exclusion radius requires a center')
        self.validate_vertex_clearance(snapshot.coordinates)
        self.collision = (CollisionGuard(snapshot.coordinates, snapshot.faces)
                          if self.options.check_self_intersections else None)

    def validate_vertex_clearance(self, candidate):
        """Check vertex endpoints and linear paths against the exclusion sphere."""
        if self.vertex_exclusion_radius == 0:
            return {}
        candidate = np.asarray(candidate, dtype=float)
        start = self.snapshot.coordinates
        if (candidate.shape != start.shape or not np.isfinite(candidate).all()
                or not np.isfinite(start).all()):
            raise ValueError('Vertex clearance requires matching finite vertex arrays')
        offset = start - self.vertex_exclusion_center
        motion = candidate - start
        length_squared = np.sum(motion * motion, axis=1)
        fraction = np.clip(np.divide(-np.sum(offset * motion, axis=1), length_squared,
                                     out=np.zeros(len(start)), where=length_squared > 0), 0, 1)
        closest = offset + fraction[:, None] * motion
        distances = np.linalg.norm(closest, axis=1)
        index = int(np.argmin(distances))
        minimum = float(distances[index])
        if minimum < self.vertex_exclusion_radius:
            raise ValueError(f'Vertex exclusion violated: vertex {index} reaches distance '
                             f'{minimum:.8g}, below {self.vertex_exclusion_radius:.8g} '
                             'along its linear update')
        return {'min_vertex_exclusion_distance': float(np.min(np.linalg.norm(
                    candidate - self.vertex_exclusion_center, axis=1))),
                'min_vertex_exclusion_path_distance': minimum}

    def collision_step_limit(self, candidate):
        if self.collision is None:
            return 1.0
        return self.collision.step_limit(self.snapshot.coordinates, candidate)

    def validate(self, candidate):
        """Raise on infeasibility; return diagnostics without changing coordinates."""
        s = self.snapshot
        candidate = np.asarray(candidate, dtype=float)
        if candidate.shape != s.coordinates.shape:
            raise ValueError('Candidate must match the snapshot vertex array')
        clearance = self.validate_vertex_clearance(candidate)
        valid, reason = check_candidate(s.coordinates, candidate, s.faces, self.options)
        if not valid:
            raise ValueError(reason)
        if not locally_injective_path(s.coordinates, candidate, s.faces, self.options.min_area_ratio):
            raise ValueError('Triangle collapse/folding along linear update')
        metrics = validate_mesh(candidate, s.faces)
        metrics.update(clearance)
        if self.volume_target is not None:
            residual = metrics['volume'] - self.volume_target
            if abs(residual) > volume_tolerance(self.volume_target, self.options):
                raise ValueError('Volume constraint violated')
            metrics['volume_residual'] = residual
        centroid, _ = volume_centroid(candidate, s.faces)
        metrics['centroid'] = centroid.tolist()
        if self.centroid_target is not None:
            residual = float(np.max(np.abs(centroid - self.centroid_target)))
            if residual > self.centroid_atol:
                raise ValueError('Centroid constraint violated')
            metrics['centroid_residual'] = residual
        if self.collision is not None:
            if self.collision.has_intersections(candidate) or self.collision_step_limit(candidate) < 1.0:
                raise ValueError('Self-intersection along linear update')
            metrics['global_intersections'] = 'checked_with_ipc'
            metrics['continuous_collisions'] = 'checked_with_ipc'
        quality = quality_metrics(s.coordinates, candidate, s.faces)
        metrics['min_area_ratio'] = float(quality['area_ratio'].min())
        metrics['min_normal_cosine'] = float(quality['normal_cosine'].min())
        return metrics

    def write(self, candidate, output_path):
        """Publish exact coordinates only after file round-trip validation.

        Returns metrics and the realized displacement read from the saved mesh.
        Existing outputs are never overwritten.
        """
        s = self.snapshot
        if Path(output_path).suffix.lower() != '.h5m':
            raise ValueError('Constraint API writes DAGMC .h5m files')
        self.validate(candidate)
        if file_hash(s.path) != s.input_sha256:
            raise ValueError('Snapshot input changed')
        mesh = MoabMesh(s.path)
        if not np.array_equal(mesh.coordinates, s.coordinates) or not np.array_equal(mesh.faces, s.faces):
            raise ValueError('Snapshot does not match input geometry')
        with staged_mesh(s.path, output_path) as temporary:
            mesh.set_coordinates(candidate)
            mesh.write(temporary)
            reloaded = MoabMesh(temporary)
            verify_roundtrip(mesh, reloaded, candidate)
            metrics = self.validate(reloaded.coordinates)
            if file_hash(s.path) != s.input_sha256:
                raise ValueError('Input changed during export')
            metrics['output_sha256'] = file_hash(temporary)
            actual = reloaded.coordinates - s.coordinates
        return metrics, actual
