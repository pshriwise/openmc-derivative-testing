"""Same-mesh, same-gradient INT experiments with a Cartesian reference."""

import csv
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np

from ..mesh.collision import CollisionGuard
from ..mesh.reporting import write_report
from .engine import mesh_metrics, optimize_vertices, validate_candidate, geometry_hash
from .fixtures import write_obj


@dataclass(frozen=True)
class DirectConfig:
    """Independent Cartesian gradient-descent baseline settings."""
    learning_rate: float = 0.005
    iterations: int = 20

    def __post_init__(self):
        if not np.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError('Direct learning rate must be positive and finite')
        if isinstance(self.iterations, bool) or not isinstance(self.iterations, int) or self.iterations < 0:
            raise ValueError('Direct iterations must be a nonnegative integer')


def compare_methods(vertices, faces, gradient, config, output_dir, *,
                    direct_config=None, provenance=None):
    """Save three methods, unscaled metrics, normalized views, and a report.

    Parameters
    ----------
    direct_config : DirectConfig, optional
        Independent step size/count for the raw Cartesian baseline.
    """
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    x, f, g = np.asarray(vertices), np.asarray(faces), np.asarray(gradient)
    direct_config = direct_config or DirectConfig()
    unpenalized = replace(config, volume_weight=0., range_weight=0.)
    ls = optimize_vertices(x, f, g, unpenalized)
    penalty = optimize_vertices(x, f, g, config)
    collision = CollisionGuard(x, f) if config.enable_safeguards and config.check_self_intersections else None
    direct, direct_status, direct_steps = x.copy(), 'completed', 0
    direction = -g if config.objective == 'minimize' else g
    for step in range(direct_config.iterations):
        candidate = direct + direct_config.learning_rate * direction
        try:
            validate_candidate(x, candidate, f, config, collision=collision, previous=direct)
        except ValueError as exc:
            direct_status = f'rejected_step: {exc}'
            break
        direct, direct_steps = candidate, step + 1
    candidates = {'direct': direct, 'large_steps': ls.vertices,
                  'large_steps_penalty': penalty.vertices}
    information = {
        'direct': dict(status=direct_status, settings=asdict(direct_config), completed_steps=direct_steps),
        'large_steps': dict(status=ls.status, settings=asdict(unpenalized), history=ls.history),
        'large_steps_penalty': dict(status=penalty.status, settings=asdict(config), history=penalty.history)}
    summary = dict(geometry_sha256=geometry_hash(x, f), provenance=provenance,
                   environment=ls.diagnostics['environment'], methods={},
                   normalized_view='Each nonzero displacement rescaled to max vertex motion 0.02 * '
                                   'bounding-box diagonal. Visualization only; constraints are not restored.')
    write_obj(output / 'initial.obj', x, f)
    np.savez_compressed(output / 'input.npz', vertices=x, faces=f, gradients=g)
    normalized = {}
    target = 0.02 * np.linalg.norm(np.ptp(x, axis=0))
    for name, candidate in candidates.items():
        if candidate.shape != x.shape or not np.isfinite(candidate).all():
            raise ValueError(f'{name} returned invalid coordinates')
        metrics = mesh_metrics(x, candidate, f, g)
        summary['methods'][name] = dict(metrics=metrics, **information[name])
        write_obj(output / f'{name}.obj', candidate, f)
        displacement = candidate - x
        scale = metrics['max_displacement']
        normalized[name] = x + displacement * (target / scale if scale > 1e-10 else 0.)
        write_obj(output / f'{name}_normalized_view.obj', normalized[name], f)
        summary['methods'][name]['normalized_view_metrics'] = mesh_metrics(x, normalized[name], f, g)
    with (output / 'summary.csv').open('x', newline='') as stream:
        keys = ['signed_volume', 'relative_volume_error', 'max_displacement', 'rms_displacement',
                'min_area_ratio', 'min_edge_length', 'max_aspect_ratio', 'descent_alignment',
                'predicted_tally_change', 'prediction_per_rms_displacement']
        writer = csv.DictWriter(stream, fieldnames=['method', 'status'] + keys)
        writer.writeheader()
        for name, info in summary['methods'].items():
            writer.writerow(dict(method=name, status=info['status'],
                                 **{k: info['metrics'][k] for k in keys}))
    plot_comparison(x, f, candidates, output / 'comparison.png')
    plot_comparison(x, f, normalized, output / 'normalized_view.png')
    write_report(output / 'summary.json', summary)
    lines = ['# INT deformation comparison', '',
             'The same fixed mesh and raw Cartesian derivative were supplied to all methods.',
             'No new transport calculation was performed. Predicted changes use the frozen derivative.', '',
             '| Method | Status | Relative volume change | Max displacement | Predicted ΔT |',
             '| --- | --- | ---: | ---: | ---: |']
    for name, info in summary['methods'].items():
        m = info['metrics']
        lines.append(f"| {name} | {info['status']} | {m['relative_volume_error']:.6g} | "
                     f"{m['max_displacement']:.6g} | {m['predicted_tally_change']:.6g} |")
    lines += ['', 'Volume penalties are soft. Differences in update size are not evidence of superiority.',
              'Compare prediction_per_rms_displacement and descent_alignment in summary.csv.',
              'comparison.png shows actual coordinates; normalized_view.png compares displacement patterns.',
              summary['normalized_view'],
              'OBJ files open in ParaView (Wavefront reader), MeshLab, or Blender.',
              'Local quality checks do not certify absence of self-intersection; see recorded collision settings.']
    (output / 'report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return summary


def plot_comparison(reference, faces, candidates, path):
    """Optional standalone figure with shared limits and displacement colors."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    except ImportError:
        return
    points = np.concatenate([reference, *candidates.values()])
    center = (points.min(axis=0) + points.max(axis=0)) / 2
    radius = np.ptp(points, axis=0).max() * .55
    maximum = max(np.linalg.norm(v - reference, axis=1).max() for v in candidates.values())
    norm = matplotlib.colors.Normalize(0, max(maximum, 1e-12))
    fig = plt.figure(figsize=(5 * len(candidates), 4), layout='constrained')
    axes = []
    for index, (name, vertices) in enumerate(candidates.items(), 1):
        axis = fig.add_subplot(1, len(candidates), index, projection='3d')
        axes.append(axis)
        displacement = np.linalg.norm(vertices - reference, axis=1)[faces].mean(axis=1)
        mesh = Poly3DCollection(vertices[faces], facecolors=plt.cm.viridis(norm(displacement)),
                                edgecolors=(0, 0, 0, .2), linewidths=.3)
        axis.add_collection3d(mesh)
        axis.set(xlim=(center[0]-radius, center[0]+radius), ylim=(center[1]-radius, center[1]+radius),
                 zlim=(center[2]-radius, center[2]+radius), title=name, xlabel='X', ylabel='Y', zlabel='Z')
        axis.set_box_aspect((1, 1, 1))
    fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap='viridis'), ax=axes,
                 shrink=.7, label='Vertex displacement (mesh units)')
    fig.savefig(path, dpi=160)
    plt.close(fig)
