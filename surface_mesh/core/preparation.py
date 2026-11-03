"""Scale a spherical DAGMC mesh to a new initial radius using validated MOAB I/O."""

import json
from pathlib import Path

import numpy as np
from surface_mesh.mesh import DeformationOptions, MeshConstraints, load_mesh

from .runtime import atomic_write

INPUT_MESH = Path(__file__).resolve().parents[1] / 'sphere.h5m'
OUTPUT_MESH = Path(__file__).resolve().parents[1] / 'results' / 'inputs/sphere_radius_2cm.h5m'
TARGET_RADIUS_CM = 2.0
LOCAL_CENTER_CM = (0., 0., 0.)


def prepare(input_mesh=INPUT_MESH, output_mesh=OUTPUT_MESH, radius=TARGET_RADIUS_CM):
    """Change the initial volume while preserving connectivity and DAGMC tags."""
    if not np.isfinite(radius) or radius <= 0:
        raise ValueError('Radius must be positive and finite')
    snapshot = load_mesh(input_mesh)
    center = np.asarray(LOCAL_CENTER_CM)
    offsets = snapshot.coordinates - center
    radii = np.linalg.norm(offsets, axis=1)
    original_radius = float(radii.mean())
    if not np.allclose(radii, original_radius, rtol=1e-10, atol=1e-10):
        raise ValueError('Input vertices must lie on a sphere about LOCAL_CENTER_CM')
    scale = radius / original_radius
    candidate = center + scale * offsets
    output_mesh = Path(output_mesh)
    output_mesh.parent.mkdir(parents=True, exist_ok=True)
    # This is preparation of a new problem; optimization subsequently conserves
    # the resulting volume. Continuous collision and metadata checks stay on.
    geometry = MeshConstraints(snapshot, options=DeformationOptions(
        preserve_volume=False, check_self_intersections=True,
        min_area_ratio=min(0.05, 0.5 * scale**2)))
    metrics, _ = geometry.write(candidate, output_mesh)
    saved = load_mesh(output_mesh)
    np.testing.assert_allclose(np.linalg.norm(saved.coordinates - center, axis=1),
                               radius, rtol=1e-12, atol=1e-12)
    report = dict(input_mesh=str(snapshot.path), input_sha256=snapshot.input_sha256,
                  output_mesh=str(output_mesh.resolve()), original_radius=original_radius,
                  target_radius=radius, scale=scale, local_center=center.tolist(),
                  **metrics)
    encoded = (json.dumps(report, indent=2) + '\n').encode()
    atomic_write(output_mesh.with_suffix('.json'), lambda stream: stream.write(encoded))
    return report
