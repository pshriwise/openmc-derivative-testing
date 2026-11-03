"""Lazy Mitsuba initialization; no renderer, scene, or substitute Laplacian."""

from importlib.metadata import PackageNotFoundError, version
import platform

import numpy as np


def initialize_backend(requested='auto'):
    """Probe an actual sparse solve and its reverse derivative, CPU first."""
    try:
        import mitsuba as mi
        import drjit as dr
        import cholespy  # noqa: F401 -- missing optional solver should fail early
    except ImportError as exc:
        raise ImportError('Install the Mitsuba, Dr.Jit and cholespy dependencies in surface_mesh/requirements.txt') from exc
    failures = []
    variants = ('llvm_ad_rgb', 'cuda_ad_rgb') if requested == 'auto' else (requested,)
    for variant in variants:
        try:
            mi.set_variant(variant)
            v = mi.Float([0., 0., 0., 1., 0., 0., 0., 1., 0.])
            ls = mi.ad.LargeSteps(v, mi.UInt([0, 1, 2]), 1.)
            u = ls.to_differential(v)
            dr.enable_grad(u)
            dr.backward(dr.sum(ls.from_differential(u)))
            if not np.isfinite(dr.grad(u).numpy()).all():
                raise RuntimeError('Nonfinite probe derivative')
            versions = {name: version(name) for name in ('mitsuba', 'drjit', 'cholespy', 'numpy')}
            for name in ('pytest', 'ipctk', 'pymoab', 'pydagmc'):
                try:
                    versions[name] = version(name)
                except PackageNotFoundError:
                    versions[name] = 'distribution version unavailable'
            return mi, dr, versions | {
                'python': platform.python_version(), 'platform': platform.platform(),
                'backend': variant, 'coordinate_precision': 'float32',
                'failed_backend_probes': failures}
        except Exception as exc:
            failures.append(f'{variant}: {exc}')
    raise RuntimeError('No working LargeSteps AD backend: ' + '; '.join(failures))


def loss_terms(mi, dr, v, v0, gradient, faces, config, initial_volume, origin):
    """Build all objective terms on the same Cartesian reconstruction graph."""
    # Unravel exactly once and use that representation for *all* branches.
    # Dr.Jit 1.5.0's permuting gathers can overwrite adjoints when flat and
    # independently unraveled branches of the same array are mixed.
    absolute = dr.unravel(mi.Point3f, v)
    points = absolute - mi.Point3f(origin)
    corners = [dr.gather(mi.Point3f, points, mi.UInt(faces[:, k])) for k in range(3)]
    volume = dr.sum(dr.dot(corners[0], dr.cross(corners[1], corners[2]))) / 6.
    reference = dr.unravel(mi.Point3f, v0)
    g = dr.unravel(mi.Point3f, mi.Float(gradient.ravel()))
    transport = dr.sum(dr.dot(g, absolute - reference))
    range_loss = mi.Float(0.)
    # INT sums over triangle corners, including repeated incident vertices.
    for axis, bound in enumerate(config.bounds):
        if bound is not None:
            values = dr.gather(mi.Float, absolute[axis], mi.UInt(faces.ravel()))
            range_loss += dr.sum(dr.square(dr.maximum(bound[0] - values, 0.))
                                 + dr.square(dr.maximum(values - bound[1], 0.)))
    volume_loss = dr.square(volume - initial_volume)
    sign = 1. if config.objective == 'minimize' else -1.
    loss = sign * transport + config.volume_weight * volume_loss + config.range_weight * range_loss
    return loss, transport, volume_loss, range_loss, volume
