"""INT's projected Botsch remeshing, with a rebuilt H5M triangulation."""

from importlib.metadata import version
from pathlib import Path
from time import perf_counter

import numpy as np

from ..mesh.geometry import mesh_arrays, signed_volume
from ..mesh.moab_adapter import MoabMesh, staged_mesh, verify_roundtrip
from ..mesh.reporting import file_hash
from ..mesh.topology import validate_mesh


def remesh_h5m(snapshot, output_path, config):
    """Replace all vertices/faces, retaining DAGMC sets, senses and material tags.

    Derivatives and optimizer state cannot be reused on the returned mesh.
    Projection is onto the incoming deformed surface, not the original sphere.
    This does not impose exact volume conservation or repair self-intersections.
    """
    if Path(output_path).suffix.lower() != '.h5m':
        raise ValueError('Remeshing requires an .h5m output')
    try:
        from gpytoolbox import remesh_botsch
    except ImportError as exc:
        raise ImportError('Remeshing requires dagmc-mesh-deform[int-remesh]') from exc
    started = perf_counter()
    x, f = snapshot.coordinates, snapshot.faces
    edges = np.concatenate((f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]))
    target = config.remesh_target_edge_length
    if target is None:
        target = float(np.linalg.norm(x[edges[:, 0]] - x[edges[:, 1]], axis=1).mean())
    vertices, faces = remesh_botsch(x, f, i=config.remesh_iterations, h=target, project=True)
    vertices, faces = mesh_arrays(vertices, faces)
    if config.enable_safeguards:
        validate_mesh(vertices, faces)
    if file_hash(snapshot.path) != snapshot.input_sha256:
        raise ValueError('Remeshing input changed')
    mesh = MoabMesh(snapshot.path)
    if not np.array_equal(mesh.coordinates, x) or not np.array_equal(mesh.faces, f):
        raise ValueError('Snapshot does not match input geometry')
    model = mesh.dagmc_model()
    surface = model.surfaces[0].handle
    # Keep the semantic DAGMC database; discard its entire old triangulation.
    # Explicitly clear memberships before deletion (MOAB meshsets need not
    # track deletions). No old derivative fields are transferred.
    old_entities = np.concatenate((mesh.vertex_handles, mesh.triangle_handles))
    for handle in mesh.mb.get_entities_by_type(0, mesh.types.MBENTITYSET):
        mesh.mb.remove_entities(handle, old_entities)
    mesh.mb.delete_entities(mesh.triangle_handles)
    mesh.mb.delete_entities(mesh.vertex_handles)
    mesh.vertex_handles = np.asarray(mesh.mb.create_vertices(
        np.ascontiguousarray(vertices).ravel()), dtype=np.uint64)
    mesh.triangle_handles = np.asarray(mesh.mb.create_elements(
        mesh.types.MBTRI, np.ascontiguousarray(mesh.vertex_handles[faces])), dtype=np.uint64)
    mesh.mb.add_entities(surface, mesh.vertex_handles)
    mesh.mb.add_entities(surface, mesh.triangle_handles)
    mesh.coordinates, mesh.faces = vertices.copy(), faces.copy()
    with staged_mesh(snapshot.path, output_path) as staged:
        mesh.write(staged)
        reloaded = MoabMesh(staged)
        verify_roundtrip(mesh, reloaded, vertices, allow_reader_defaults=True)
        if reloaded.inspect_dagmc() != dict(surface_id=snapshot.surface_id, volume_id=snapshot.volume_id):
            raise ValueError('DAGMC IDs changed during remeshing')
        if file_hash(snapshot.path) != snapshot.input_sha256:
            raise ValueError('Remeshing input changed during export')
        digest = file_hash(staged)
    before, after = signed_volume(x, f), signed_volume(vertices, faces)
    return dict(method='gpytoolbox.remesh_botsch', version=version('gpytoolbox'),
                iterations=config.remesh_iterations, project=True, target_edge_length=target,
                vertices_before=len(x), vertices_after=len(vertices),
                faces_before=len(f), faces_after=len(faces),
                volume_before=before, volume_after=after, volume_change=after - before,
                output_sha256=digest, elapsed_seconds=perf_counter() - started,
                derivative_transfer='none; fresh transport required')
