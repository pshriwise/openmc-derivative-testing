"""INT-style LargeSteps/Adam with a local externally differentiated objective."""

from dataclasses import asdict, dataclass
import hashlib
from time import perf_counter

import numpy as np

from ..mesh.collision import CollisionGuard, locally_injective_path
from ..mesh.geometry import mesh_arrays, quality_metrics, signed_volume, volume_gradient
from ..mesh.topology import validate_mesh
from .backend import initialize_backend, loss_terms
from .config import IntConfig


@dataclass
class OptimizationResult:
    vertices: np.ndarray
    faces: np.ndarray
    history: list
    diagnostics: dict
    status: str
    adam_state: dict | None = None


def transfer_adam_state(state, vertices):
    """Map saved Adam moments onto new vertices by nearest saved vertex.

    Moments live in LargeSteps differential coordinates, one XYZ row per
    vertex. Identical vertex sets map exactly; after remeshing this is a
    nearest-neighbor approximation, not a conservative transfer.
    """
    from scipy.spatial import cKDTree
    source = np.asarray(state['vertices'], dtype=float)
    m, v = np.asarray(state['m'], dtype=float), np.asarray(state['v'], dtype=float)
    if m.shape != source.shape or v.shape != source.shape or np.any(v < 0):
        raise ValueError('Adam state must hold (N, 3) moments with nonnegative second moments')
    distance, index = cKDTree(source).query(np.asarray(vertices, dtype=float))
    return m[index], v[index], int(state['t']), float(distance.max())


def geometry_hash(vertices, faces):
    """Hash ordered float64 XYZ and int64 connectivity, including their shapes."""
    digest = hashlib.sha256()
    for value, dtype in ((vertices, '<f8'), (faces, '<i8')):
        array = np.asarray(value, dtype=dtype)
        digest.update(str(array.shape).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def mesh_metrics(reference, candidate, faces, gradient=None):
    """Report volume, geometry quality, motion, and raw predicted tally change."""
    quality = quality_metrics(reference, candidate, faces)
    edges = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    lengths = np.linalg.norm(candidate[edges[:, 0]] - candidate[edges[:, 1]], axis=1)
    initial_lengths = np.linalg.norm(reference[edges[:, 0]] - reference[edges[:, 1]], axis=1)
    delta = candidate - reference
    translation = delta.mean(axis=0)
    volume, initial = signed_volume(candidate, faces), signed_volume(reference, faces)
    # sqrt(3) * longest edge squared / (4 * area): 1 for an equilateral triangle.
    longest = lengths.reshape(3, -1).max(axis=0)
    result = dict(signed_volume=volume, absolute_volume=abs(volume),
                  volume_change=volume - initial, relative_volume_error=(volume - initial) / initial,
                  min_area_ratio=float(quality['area_ratio'].min()),
                  max_area_ratio=float(quality['area_ratio'].max()),
                  min_normal_cosine=float(quality['normal_cosine'].min()),
                  min_edge_length=float(lengths.min()),
                  min_edge_ratio=float((lengths / initial_lengths).min()),
                  max_aspect_ratio=float((np.sqrt(3) * longest**2 / (4 * quality['areas'])).max()),
                  max_displacement=float(np.linalg.norm(delta, axis=1).max()),
                  rms_displacement=float(np.sqrt(np.mean(np.sum(delta**2, axis=1)))),
                  displacement_norm=float(np.linalg.norm(delta)),
                  mean_displacement=translation.tolist(),
                  max_displacement_without_translation=float(np.linalg.norm(delta - translation, axis=1).max()),
                  bounds=[candidate.min(axis=0).tolist(), candidate.max(axis=0).tolist()])
    if gradient is not None:
        prediction = float(np.sum(gradient * delta))
        denominator = float(np.linalg.norm(gradient) * np.linalg.norm(delta))
        result.update(predicted_tally_change=prediction,
                      descent_alignment=None if denominator == 0 else -prediction / denominator,
                      prediction_per_rms_displacement=(prediction / result['rms_displacement']
                                                       if result['rms_displacement'] else None))
    return result


def validate_candidate(reference, candidate, faces, config, *, collision=None, previous=None):
    """Reject an unsafe step; never project, smooth, clip, or repair positions."""
    if candidate.shape != reference.shape or not np.isfinite(candidate).all():
        raise ValueError('Nonfinite or mismatched candidate coordinates')
    metrics = mesh_metrics(reference, candidate, faces)
    if not config.enable_safeguards:
        return metrics
    if metrics['signed_volume'] <= 0:
        raise ValueError('Signed-volume orientation changed')
    for key, threshold in (('min_area_ratio', config.min_area_ratio),
                           ('min_normal_cosine', config.min_normal_cosine),
                           ('min_edge_ratio', config.min_edge_ratio)):
        if not np.isfinite(metrics[key]) or metrics[key] < threshold:
            raise ValueError(f'Quality check failed: {key}')
    if metrics['max_displacement'] > config.max_displacement * (1 + 1e-12):
        raise ValueError('Frozen-gradient displacement limit exceeded')
    starts = (reference,) if previous is None else (reference, previous)
    for start in starts:
        if not locally_injective_path(start, candidate, faces, config.min_area_ratio):
            raise ValueError('Triangle collapse/folding along update')
        if collision is not None and collision.step_limit(start, candidate) < 1.:
            raise ValueError('Continuous self-collision along update')
    if collision is not None and collision.has_intersections(candidate):
        raise ValueError('Candidate self-intersection')
    return metrics


def optimize_vertices(vertices, faces, gradient, config=None, *, volume_target=None, adam_state=None):
    """Optimize a frozen linear tally surrogate plus INT-style soft penalties.

    Parameters
    ----------
    vertices, faces : array-like
        Outward oriented isolated shell, (N, 3) XYZ and (M, 3) integer indices.
    gradient : array-like
        Raw dT/dXYZ, (N, 3), evaluated at exactly these ordered vertices.
    config : IntConfig, optional
        Independent settings. Rejected steps stop at the last valid iterate.
    volume_target : float, optional
        Fixed soft-penalty target across outer updates/remeshes. Defaults to
        the input mesh volume when this function is used on its own.
    adam_state : dict, optional
        A previous result's ``adam_state`` (vertices, m, v, t). Its moments
        are transferred onto these vertices so momentum and step scaling
        continue across outer updates. Defaults to a fresh optimizer.

    Returns
    -------
    OptimizationResult
        Post-update positions, unchanged faces, history, provenance, and the
        final Adam state at the returned positions.
        No nonlinear transport response is evaluated here.
    """
    started = perf_counter()
    config = IntConfig() if config is None else config
    if not isinstance(config, IntConfig):
        raise TypeError('This engine requires IntConfig, not legacy settings')
    x, f = mesh_arrays(vertices, faces)
    x, f = x.copy(), f.copy()
    if config.enable_safeguards:
        topology = validate_mesh(x, f)
        if topology['unused_vertices'] or topology['coincident_vertex_count']:
            raise ValueError('Require used, distinct vertices; LargeSteps merges coincident positions')
    g = np.asarray(gradient, dtype=float)
    if g.shape != x.shape or not np.isfinite(g).all():
        raise ValueError('Expected finite (N, 3) Cartesian gradient')
    mi, dr, environment = initialize_backend(config.backend)
    if not np.isfinite(x.astype(np.float32)).all() or not np.isfinite(g.astype(np.float32)).all():
        raise ValueError('Coordinates/gradient exceed the backend float32 range')
    if config.enable_safeguards and len(np.unique(x.astype(np.float32), axis=0)) != len(x):
        raise ValueError('Distinct vertices coincide in float32 backend precision')
    v0 = mi.Float(x.ravel())
    ls = mi.ad.LargeSteps(v0, mi.UInt(f.ravel()), config.large_steps_lambda)
    opt = mi.ad.Adam(lr=config.learning_rate)
    opt['u'] = ls.to_differential(v0)
    # LargeSteps stores u (and so Adam's moments) for unique vertices sorted by
    # position: row k belongs to vertex unique_vertex[k], and vertex i to row
    # vertex_row[i]. Saved moments use input vertex order.
    unique_vertex = np.asarray(ls.index.numpy(), dtype=np.int64)
    vertex_row = np.asarray(ls.inverse.numpy(), dtype=np.int64)
    transfer = None
    if adam_state is not None:
        m, v, t, distance = transfer_adam_state(adam_state, x)
        m, v = m[unique_vertex], v[unique_vertex]
        value, promoted, lr, (_, _, _, v_max) = opt.state['u']
        opt.state['u'] = value, promoted, lr, (mi.UInt(t), mi.Float(m.ravel()), mi.Float(v.ravel()), v_max)
        transfer = dict(previous_steps=t, previous_vertices=len(adam_state['vertices']),
                        max_transfer_distance=distance)
    reconstructed = ls.from_differential(opt['u']).numpy().reshape(-1, 3).astype(float)
    error = float(np.max(np.abs(reconstructed - x)))
    if config.enable_safeguards and (not np.isfinite(error) or error > 2e-5 * max(1., float(np.max(np.abs(x))))):
        raise ValueError(f'LargeSteps round-trip error too large: {error}')
    initial_volume = signed_volume(x, f) if volume_target is None else float(volume_target)
    origin = (x.min(axis=0) / 2 + x.max(axis=0) / 2).tolist()
    collision = CollisionGuard(x, f) if config.enable_safeguards and config.check_self_intersections else None
    diagnostics = dict(config=asdict(config), environment=environment,
                       geometry_sha256=geometry_hash(x, f), initial_volume=initial_volume,
                       roundtrip_max_error=error, gradient_norm=float(np.linalg.norm(g)),
                       gradient_nonzero_components=int(np.count_nonzero(g)),
                       range_bounds_active=any(b is not None for b in config.bounds),
                       adam_state_transfer=transfer,
                       self_intersections='checked_with_ipc' if collision else 'unverified',
                       surrogate='frozen local linear approximation; not an updated transport response')
    history, current, status = [], x.copy(), 'completed'
    range_violated = any(b is not None and np.any((x[:, k] < b[0]) | (x[:, k] > b[1]))
                         for k, b in enumerate(config.bounds))
    # Avoid moving a stationary mesh solely due to sparse-solve roundoff.
    if (not np.any(g) and not (config.range_weight and range_violated)
            and not (config.volume_weight and volume_target is not None
                     and signed_volume(x, f) != initial_volume)):
        status = 'stationary'
    else:
        for step in range(1, config.iterations + 1):
            v = ls.from_differential(opt['u'])
            terms = loss_terms(mi, dr, v, v0, g, f, config, initial_volume, origin)
            dr.backward(terms[0])
            gu = dr.grad(opt['u']).numpy()
            if not np.isfinite(gu).all():
                raise ValueError('Nonfinite differential-coordinate gradient')
            opt.step()
            post = ls.from_differential(opt['u'])
            candidate = post.numpy().reshape(-1, 3).astype(float)
            entry = dict(step=step, differential_gradient_norm=float(np.linalg.norm(gu)))
            try:
                validate_candidate(x, candidate, f, config, collision=collision, previous=current)
                post_terms = loss_terms(mi, dr, post, v0, g, f, config, initial_volume, origin)
                values = [float(term.numpy()[0]) for term in post_terms]
                if not np.isfinite(values).all():
                    raise ValueError('Nonfinite objective or penalty')
                entry.update(zip(('loss', 'transport_surrogate_change', 'volume_penalty',
                                  'range_penalty', 'ad_signed_volume'), values))
                entry.update(weighted_volume_penalty=config.volume_weight * values[2],
                             weighted_range_penalty=config.range_weight * values[3])
                entry.update(mesh_metrics(x, candidate, f, g), accepted=True)
                current = candidate
            except ValueError as exc:
                entry.update(accepted=False, reason=str(exc))
                status = 'rejected_step'
            history.append(entry)
            if status == 'rejected_step':
                break
    diagnostics['final'] = mesh_metrics(x, current, f, g)
    penalty_gradient = (2 * config.volume_weight * (signed_volume(current, f) - initial_volume)
                        * volume_gradient(current, f))
    diagnostics['final']['volume_to_transport_gradient_norm_ratio'] = (
        float(np.linalg.norm(penalty_gradient) / np.linalg.norm(g)) if np.any(g) else None)
    t, m, v, _ = opt.state['u'][3]
    adam = dict(vertices=current.copy(), m=m.numpy().reshape(-1, 3).astype(float)[vertex_row],
                v=v.numpy().reshape(-1, 3).astype(float)[vertex_row], t=int(np.asarray(t.numpy()).ravel()[0]))
    diagnostics['elapsed_seconds'] = perf_counter() - started
    return OptimizationResult(current, f, history, diagnostics, status, adam)
