"""Checks for one connected, outward-oriented closed triangle shell."""

from collections import defaultdict

import numpy as np

from .geometry import mesh_arrays, signed_volume, triangle_areas


def validate_mesh(coordinates, faces):
    x, f = mesh_arrays(coordinates, faces)
    if len(np.unique(np.sort(f, axis=1), axis=0)) != len(f):
        raise ValueError("Duplicate triangles")
    if np.any(np.diff(np.sort(f, axis=1), axis=1) == 0):
        raise ValueError("Triangle repeats a vertex")
    areas = triangle_areas(x, f)
    if not np.isfinite(areas).all() or np.any(areas <= 0):
        raise ValueError("Degenerate or nonfinite triangle area")

    edge_faces = defaultdict(list)
    links = defaultdict(list)
    for face_index, (a, b, c) in enumerate(f):
        for u, v in ((a, b), (b, c), (c, a)):
            edge_faces[tuple(sorted((int(u), int(v))))].append((face_index, 1 if u < v else -1))
        links[int(a)].append((int(b), int(c)))
        links[int(b)].append((int(c), int(a)))
        links[int(c)].append((int(a), int(b)))
    bad_edges = sum(len(entries) != 2 for entries in edge_faces.values())
    if bad_edges:
        raise ValueError(f"Open or nonmanifold boundary: {bad_edges} edges do not have two faces")
    if any(entries[0][1] == entries[1][1] for entries in edge_faces.values()):
        raise ValueError("Inconsistent triangle winding")

    adjacent = defaultdict(set)
    for entries in edge_faces.values():
        a, b = entries[0][0], entries[1][0]
        adjacent[a].add(b)
        adjacent[b].add(a)
    if len(_reachable(adjacent, 0)) != len(f):
        raise ValueError("Expected one connected triangle boundary")
    # Edge-manifoldness alone does not detect a pinched vertex with multiple fans.
    for vertex, edges in links.items():
        graph = defaultdict(set)
        for a, b in edges:
            graph[a].add(b)
            graph[b].add(a)
        if any(len(n) != 2 for n in graph.values()) or len(_reachable(graph, next(iter(graph)))) != len(graph):
            raise ValueError(f"Nonmanifold vertex link at dense vertex {vertex}")

    volume = signed_volume(x, f)
    if not np.isfinite(volume) or volume <= 0:
        raise ValueError("Boundary must have positive signed volume (outward winding)")
    used = np.unique(f)
    return {
        "vertices": len(x),
        "triangles": len(f),
        "unused_vertices": len(x) - len(used),
        "coincident_vertex_count": len(x) - len(np.unique(x, axis=0)),
        "min_area": float(areas.min()),
        "volume": volume,
        "topology": "closed_connected_oriented",
        "global_intersections": "unverified",
    }


def _reachable(graph, start):
    seen = {start}
    pending = [start]
    while pending:
        for node in graph[pending.pop()]:
            if node not in seen:
                seen.add(node)
                pending.append(node)
    return seen
