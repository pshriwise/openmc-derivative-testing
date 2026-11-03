"""The sole owner of MOAB handles, database edits, and mesh serialization."""

from contextlib import contextmanager
import os
from pathlib import Path
import tempfile

import numpy as np

from .geometry import mesh_arrays


def moab_modules():
    try:
        from pymoab import core, types
    except ImportError as exc:
        raise ImportError("Mesh I/O requires a compatible MOAB build with PyMOAB enabled") from exc
    return core, types


class MoabMesh:
    def __init__(self, path):
        core, self.types = moab_modules()
        self.path = Path(path)
        if self.path.suffix.lower() not in {".vtk", ".stl", ".h5m"}:
            raise ValueError("Supported inputs: legacy .vtk, .stl, .h5m")
        self.mb = core.Core()
        self.mb.load_file(str(self.path))
        t = self.types
        # This exact root query defines the optimizer's row ordering.
        self.vertex_handles = np.asarray(list(self.mb.get_entities_by_type(0, t.MBVERTEX)), dtype=np.uint64)
        self.triangle_handles = np.asarray(list(self.mb.get_entities_by_type(0, t.MBTRI)), dtype=np.uint64)
        for entity_type in range(t.MBEDGE, t.MBENTITYSET):
            if entity_type != t.MBTRI and len(self.mb.get_entities_by_type(0, entity_type)):
                raise ValueError("Initial prototype supports triangle-only mesh elements")
        self.coordinates = np.asarray(self.mb.get_coords(self.vertex_handles)).reshape(-1, 3).copy()
        handle_to_index = {int(h): i for i, h in enumerate(self.vertex_handles)}
        conn = self.mb.get_connectivity(self.triangle_handles)
        self.faces = np.array([handle_to_index[int(h)] for h in conn], dtype=np.intp).reshape(-1, 3)
        mesh_arrays(self.coordinates, self.faces)

    def set_coordinates(self, coordinates):
        x = np.asarray(coordinates, dtype=np.float64)
        if x.shape != self.coordinates.shape or not np.isfinite(x).all():
            raise ValueError("Coordinates must be finite and match the vertex read array")
        # Installed PyMOAB validates len(coords) == 3*N, so flatten explicitly.
        self.mb.set_coords(self.vertex_handles, np.ascontiguousarray(x).ravel())

    def add_field(self, name, values, *, association="point"):
        values = np.asarray(values)
        handles = self.vertex_handles if association == "point" else self.triangle_handles
        if association not in {"point", "cell"}:
            raise ValueError("Field association must be point or cell")
        if values.ndim not in (1, 2) or len(values) != len(handles) or not np.isfinite(values).all():
            raise ValueError("Field shape/values do not match the mesh")
        components = 1 if values.ndim == 1 else values.shape[1]
        if components < 1:
            raise ValueError("Field must have at least one component")
        integer = values.dtype.kind in "iu"
        if integer and (np.any(values < np.iinfo(np.int32).min) or np.any(values > np.iinfo(np.int32).max)):
            raise ValueError("MOAB integer visualization tags require int32 values")
        dtype = self.types.MB_TYPE_INTEGER if integer else self.types.MB_TYPE_DOUBLE
        data = np.ascontiguousarray(values, dtype=np.int32 if integer else np.float64)
        try:
            tag = self.mb.tag_get_handle(name)
        except RuntimeError:
            tag = self.mb.tag_get_handle(name, components, dtype, self.types.MB_TAG_SPARSE, create_if_missing=True)
        if tag.get_length() != components or tag.get_data_type() != dtype:
            raise ValueError(f"Existing field has incompatible schema: {name}")
        self.mb.tag_set_data(tag, handles, data)

    def read_field(self, name, *, association="point"):
        handles = self.vertex_handles if association == "point" else self.triangle_handles
        tag = self.mb.tag_get_handle(name)
        return self.mb.tag_get_data(tag, handles).copy()

    def dagmc_model(self):
        """Inspect existing topology, without silently adding missing tags."""
        import pydagmc

        before = self.inventory()
        for name in ("CATEGORY", "GEOM_DIMENSION", "GLOBAL_ID", "GEOM_SENSE_2", "NAME"):
            # NAME is optional for an ungrouped model; do not access that property
            # unless groups exist. Model's constructor only needs CATEGORY/GLOBAL_ID.
            if name != "NAME":
                self.mb.tag_get_handle(name)
        model = pydagmc.Model(self.mb)
        if model.mb is not self.mb or self.inventory() != before:
            raise ValueError("PyDAGMC inspection changed the original database metadata")
        if len(model.volumes) != 1 or len(model.surfaces) != 1:
            raise ValueError("Expected one DAGMC volume and one surface")
        volume, surface = model.volumes[0], model.surfaces[0]
        if surface.senses != [volume, None] or volume.surfaces != [surface]:
            raise ValueError("Expected outward surface with forward volume and null reverse sense")
        if not np.array_equal(np.asarray(surface.triangle_handles), self.triangle_handles):
            raise ValueError("DAGMC surface must contain every triangle exactly once")
        return model

    def inspect_dagmc(self):
        """Read single-volume topology directly, without PyDAGMC tag repair.

        Legacy material groups may omit GEOM_DIMENSION=4. That does not make
        their material assignment invalid, and must not trigger metadata edits.
        """
        mb, t = self.mb, self.types
        dim = mb.tag_get_handle('GEOM_DIMENSION')
        gid = mb.tag_get_handle('GLOBAL_ID')
        category = mb.tag_get_handle('CATEGORY')
        sense = mb.tag_get_handle('GEOM_SENSE_2')
        sets = list(mb.get_entities_by_type(0, t.MBENTITYSET))
        by_dim = {d: [h for h in sets if int(mb.tag_get_data(dim, [h], flat=True)[0]) == d]
                  for d in (2, 3)}
        if len(by_dim[2]) != 1 or len(by_dim[3]) != 1:
            raise ValueError('Expected one DAGMC volume and one surface')
        surface, volume = by_dim[2][0], by_dim[3][0]
        for handle, expected in ((surface, 'Surface'), (volume, 'Volume')):
            value = mb.tag_get_data(category, [handle], flat=True)[0]
            value = value.decode() if isinstance(value, bytes) else str(value)
            if value.rstrip('\x00') != expected:
                raise ValueError('Inconsistent DAGMC category and dimension')
        if list(mb.tag_get_data(sense, [surface], flat=True)) != [volume, 0]:
            raise ValueError('Expected forward volume and null reverse sense')
        if list(mb.get_child_meshsets(volume)) != [surface]:
            raise ValueError('Expected one surface child of the volume')
        if not np.array_equal(mb.get_entities_by_type(surface, t.MBTRI), self.triangle_handles):
            raise ValueError('DAGMC surface must contain every triangle')
        return {'surface_id': int(mb.tag_get_data(gid, [surface], flat=True)[0]),
                'volume_id': int(mb.tag_get_data(gid, [volume], flat=True)[0])}

    def inventory(self):
        """Compare entities/tags/sets using type-local read indices, not raw handles.

        Conservative: unexpected reordering is rejected rather than guessed. This
        is not a general graph-isomorphism algorithm for arbitrary MOAB databases.
        """
        mb, t = self.mb, self.types
        keys = {0: ("root", 0)}
        ordered = [np.uint64(0)]
        for kind in range(t.MBMAXTYPE):
            handles = list(mb.get_entities_by_type(0, kind))
            ordered.extend(handles)
            keys.update({int(h): (kind, i) for i, h in enumerate(handles)})
        result = []
        for handle in ordered:
            h = int(handle)
            tags = []
            for tag in mb.tag_get_tags_on_entity(handle):
                values = mb.tag_get_data(tag, np.array([handle], dtype=np.uint64), flat=True)
                if tag.get_data_type() == t.MB_TYPE_HANDLE:
                    data = tuple(keys[int(value)] for value in values)
                else:
                    data = (values.dtype.str, values.tobytes()) if values.dtype.kind not in "OU" else tuple(str(v) for v in values)
                tags.append((tag.get_name(), tag.get_data_type(), tag.get_length(), data))
            record = [keys[h], tuple(sorted(tags))]
            if h and mb.type_from_handle(handle) == t.MBENTITYSET:
                record.extend([
                    tuple(keys[int(v)] for v in mb.get_entities_by_handle(handle, as_list=True)),
                    tuple(sorted(keys[int(v)] for v in mb.get_child_meshsets(handle))),
                    tuple(sorted(keys[int(v)] for v in mb.get_parent_meshsets(handle))),
                ])
            result.append(tuple(record))
        return tuple(result)

    def reject_cached_geometry(self):
        for entity in [np.uint64(0), *self.mb.get_entities_by_type(0, self.types.MBENTITYSET)]:
            for tag in self.mb.tag_get_tags_on_entity(entity):
                if "OBB" in tag.get_name().upper():
                    raise NotImplementedError("Stored OBB structures require a verified rebuild path before deformation")

    def reject_external_payloads(self):
        try:
            import h5py
        except ImportError as exc:
            raise NotImplementedError("Editing .h5m requires h5py to check for non-MOAB HDF5 payloads") from exc
        with h5py.File(self.path, "r") as database:
            if set(database.keys()) != {"tstt"} or len(database.attrs):
                raise NotImplementedError("Non-MOAB HDF5 payload/attributes require a preservation path before editing")

    def write(self, path):
        if Path(path).suffix.lower() == ".h5m":
            self.last_write_inventory = self.inventory()
        self.mb.write_file(str(path))


def check_destination(input_path, output_path):
    source, target = Path(input_path), Path(output_path)
    if source.resolve() == target.resolve() or (target.exists() and os.path.samefile(source, target)):
        raise ValueError("Output aliases the input")
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"Refusing to replace existing output: {target}")
    if target.suffix.lower() not in {".vtk", ".h5m"}:
        raise ValueError("Outputs must be .vtk or .h5m; STL cannot preserve vertex indexing")
    target.parent.mkdir(parents=True, exist_ok=True)
    return target


@contextmanager
def staged_mesh(input_path, output_path):
    target = check_destination(input_path, output_path)
    with tempfile.TemporaryDirectory(prefix=".mesh-stage-", dir=target.parent) as directory:
        staged = Path(directory) / target.name
        yield staged
        # Hard-link publication is atomic and refuses a racing existing destination.
        os.link(staged, target)


def verify_roundtrip(original, reloaded, expected, *, allow_reader_defaults=False):
    if not np.array_equal(original.faces, reloaded.faces):
        raise ValueError("Connectivity/order changed during serialization")
    if original.coordinates.shape != reloaded.coordinates.shape:
        raise ValueError("Vertex count changed during serialization")
    tolerance = 0.0 if reloaded.path.suffix.lower() == ".h5m" else 1e-9 * max(1.0, float(np.abs(expected).max()))
    if not np.allclose(expected, reloaded.coordinates, rtol=0, atol=tolerance):
        raise ValueError("Coordinates changed beyond serialization tolerance")
    if reloaded.path.suffix.lower() == ".h5m":
        before = getattr(original, "last_write_inventory", None)
        if before is None:
            before = original.inventory()
        after = reloaded.inventory()
        if allow_reader_defaults:
            # The H5M reader supplies these conventional root defaults even for
            # a raw database that had no root tags. Permit only absent -> -1.
            defaults = {"DIRICHLET_SET", "NEUMANN_SET", "MATERIAL_SET", "GEOM_DIMENSION", "GLOBAL_ID"}
            sentinel = (np.dtype(np.int32).str, np.array([-1], dtype=np.int32).tobytes())
            normalized = []
            for old, new in zip(before, after):
                names = {tag[0] for tag in old[1]}
                tags = tuple(tag for tag in new[1] if not (
                    tag[0] not in names and tag[0] in defaults
                    and tag[1:3] == (original.types.MB_TYPE_INTEGER, 1) and tag[3] == sentinel
                ))
                normalized.append((new[0], tags, *new[2:]))
            if len(before) == len(after):
                after = tuple(normalized)
        if before != after:
            raise ValueError("MOAB metadata or relationships changed during serialization")
    return float(np.abs(expected - reloaded.coordinates).max())
