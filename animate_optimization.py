"""Animate optimization geometry and tally/derivative histories by run number."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter, writers
from matplotlib.ticker import MaxNLocator
import numpy as np

from animate_parameter_sweep import padded_limits
from config import SURFACE_PARAMETER_NAME, SURFACE_PARAMETER_UNITS
from plot_parameter_sweep import load_sweep


def history_coordinates(results):
    """Normalize scalar and vector histories and recover saved coordinate names."""
    parameters = np.asarray(results['parameter'])
    derivatives = np.asarray(results['geometry_derivative'])
    if parameters.ndim == 1:
        parameters = parameters[:, None]
    if derivatives.ndim == 1:
        derivatives = derivatives[:, None]
    if parameters.ndim != 2 or derivatives.shape != parameters.shape:
        raise ValueError('Parameter and derivative histories must have matching coordinate shapes.')
    if 'parameter_names' in results.dtype.names:
        names = tuple(np.atleast_1d(results['parameter_names'][0]))
        if not np.all(results['parameter_names'] == results['parameter_names'][0]):
            raise ValueError('Parameter names must remain consistent across evaluations.')
    elif parameters.shape[1] == 1:
        names = (SURFACE_PARAMETER_NAME,)
    else:
        raise ValueError('Vector histories must include saved parameter_names.')
    if len(names) != parameters.shape[1] or len(set(names)) != len(names) or not all(names):
        raise ValueError('Expected one unique name per optimization coordinate.')
    return parameters, derivatives, names


def path_derivatives(parameters, mean, derivatives):
    """Backward secants per distance traveled; duplicate points are undefined.

    Each segment uses its own direction, including backtracking. The projection
    uses the gradient at the start of that segment, rather than a partial
    derivative with respect to any one coordinate.
    """
    steps = np.diff(parameters, axis=0)
    lengths = np.linalg.norm(steps, axis=1)
    finite_difference = np.full(len(mean), np.nan)
    projection = np.full(len(mean), np.nan)
    moved = lengths > 0
    finite_difference[1:] = np.divide(np.diff(mean), lengths,
                                     out=np.full_like(lengths, np.nan), where=moved)
    projection[1:] = np.divide(np.sum(derivatives[:-1] * steps, axis=1), lengths,
                              out=np.full_like(lengths, np.nan), where=moved)
    return finite_difference, projection


def optimization_image_paths(plot_dir, parameters, names):
    """Read named vector image files, with support for older scalar filenames."""
    paths = []
    for index, values in enumerate(parameters):
        matches = list(Path(plot_dir).glob(f'run_{index:04d}_parameter_*.png'))
        if len(matches) != 1:
            raise ValueError(f'Expected one geometry image for run {index} in {plot_dir}; found {len(matches)}.')
        path = matches[0]
        suffix = path.stem.split('_parameter_', 1)[1]
        if '=' in suffix:
            saved = dict(part.split('=', 1) for part in suffix.split('__'))
            if set(saved) != set(names) or any(
                not np.isclose(float(saved[name]), value, rtol=1e-7, atol=1e-12)
                for name, value in zip(names, values)
            ):
                raise ValueError(f'{path} does not match the saved optimization parameters.')
        elif SURFACE_PARAMETER_NAME in names:
            # Earlier geometry plots encoded only the configured sweep parameter.
            value = values[names.index(SURFACE_PARAMETER_NAME)]
            if not np.isclose(float(suffix), value, rtol=1e-7, atol=1e-12):
                raise ValueError(f'{path} does not match {SURFACE_PARAMETER_NAME}={value}.')
        paths.append(path)
    return paths


def create_animation(input_path, plot_dir, fps=5):
    """Animate all coordinates in evaluation order, including line-search runs."""
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError('fps must be finite and positive.')
    results = load_sweep(input_path)
    if results.ndim != 1 or len(results) == 0:
        raise ValueError('Optimization history must be a nonempty one-dimensional array.')
    parameters, derivatives, names = history_coordinates(results)
    mean = results['tally_mean']
    if not all(np.all(np.isfinite(values)) for values in (parameters, mean, derivatives)):
        raise ValueError('Parameters, tally mean, and derivatives must be finite.')
    image_paths = optimization_image_paths(plot_dir, parameters, names)
    finite_difference, projection = path_derivatives(parameters, mean, derivatives)
    runs = np.arange(len(results))

    fig = plt.figure(figsize=(14, 12), layout='constrained')
    grid = fig.add_gridspec(4, 2, width_ratios=(1.3, 1))
    geometry_axis = fig.add_subplot(grid[:, 0])
    image = geometry_axis.imshow(plt.imread(image_paths[0]))
    geometry_axis.set_axis_off()
    mean_axis = fig.add_subplot(grid[0, 1])
    derivative_axis = fig.add_subplot(grid[1, 1], sharex=mean_axis)
    path_axis = fig.add_subplot(grid[2, 1], sharex=mean_axis)
    parameter_axis = fig.add_subplot(grid[3, 1], sharex=mean_axis)
    parameter_lines = [parameter_axis.plot([], [], 'o-', markersize=4, label=name)[0]
                       for name in names]
    mean_line, = mean_axis.plot([], [], 'o-', markersize=4, color='tab:blue')
    derivative_lines = [derivative_axis.plot([], [], 'o-', markersize=4,
                                             label=f'd(mean)/d({name})')[0] for name in names]
    difference_line, = path_axis.plot([], [], 'o-', markersize=4, label='Path finite difference')
    projection_line, = path_axis.plot([], [], 'o--', markersize=3, label='Previous gradient projected onto step')
    mean_axis.set_title(f'Tally mean (point labels: parameters in {SURFACE_PARAMETER_UNITS})')
    derivative_axis.set_title('Partial geometry derivatives')
    path_axis.set_title('Change per distance along optimization path')
    mean_axis.set_ylabel('Tally mean')
    derivative_axis.set_ylabel(f'Partial derivative [1/{SURFACE_PARAMETER_UNITS}]')
    path_axis.set_ylabel(f'Path derivative [1/{SURFACE_PARAMETER_UNITS}]')
    parameter_axis.set_title('Optimization parameter values')
    parameter_axis.set_ylabel(f'Parameter value [{SURFACE_PARAMETER_UNITS}]')
    parameter_axis.set_xlabel('Evaluation / run number')
    parameter_axis.legend(fontsize=8)
    derivative_axis.legend(fontsize=8)
    path_axis.legend(fontsize=8)
    pending = path_axis.text(.5, .5, 'Requires a nonzero parameter step',
                             transform=path_axis.transAxes, ha='center')
    mean_axis.tick_params(labelbottom=False)
    derivative_axis.tick_params(labelbottom=False)
    path_axis.tick_params(labelbottom=False)
    cursors = []
    path_values = np.concatenate((finite_difference, projection))
    path_values = path_values[np.isfinite(path_values)]
    for axis, values in ((mean_axis, mean), (derivative_axis, derivatives),
                         (parameter_axis, parameters),
                         (path_axis, path_values if path_values.size else np.array([0.]))):
        axis.set_xlim(-0.5, max(len(results) - 0.5, 0.5))
        axis.set_ylim(*padded_limits(values))
        axis.xaxis.set_major_locator(MaxNLocator(integer=True))
        axis.grid(True, alpha=0.3)
        cursors.append(axis.axvline(0, color='tab:red', linestyle='--', alpha=0.6))
    labels = []
    for index, value in enumerate(mean):
        label = '\n'.join(f'{name}={coordinate:.4g}' for name, coordinate in zip(names, parameters[index]))
        above = index % 2 == 0
        labels.append(mean_axis.annotate(
            label, xy=(index, value), xytext=(0, 8 if above else -10),
            textcoords='offset points', ha='center', va='bottom' if above else 'top',
            fontsize=7, visible=False,
        ))
    title = fig.suptitle('')

    def update(index):
        count = index + 1
        image.set_data(plt.imread(image_paths[index]))
        mean_line.set_data(runs[:count], mean[:count])
        for column, line in enumerate(derivative_lines):
            line.set_data(runs[:count], derivatives[:count, column])
        for column, line in enumerate(parameter_lines):
            line.set_data(runs[:count], parameters[:count, column])
        difference_line.set_data(runs[:count], finite_difference[:count])
        projection_line.set_data(runs[:count], projection[:count])
        pending.set_visible(not np.any(np.isfinite(finite_difference[:count])))
        for label_index, label in enumerate(labels):
            label.set_visible(label_index < count)
        for cursor in cursors:
            cursor.set_xdata([index, index])
        coordinates = ', '.join(f'{name}={value:.8g}' for name, value in zip(names, parameters[index]))
        title.set_text(f'Optimization run {index} ({count}/{len(results)}) — {coordinates} {SURFACE_PARAMETER_UNITS}')
        return image, mean_line, *derivative_lines, *parameter_lines, difference_line, projection_line, pending, title, *cursors, *labels

    return FuncAnimation(
        fig, update, frames=len(results), interval=1000 / fps,
        blit=False, repeat=True, cache_frame_data=False,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('filename', nargs='?', default='optimization_history.npy')
    parser.add_argument('--plot-dir', default='optimization_geom',
                        help='Directory of run-indexed geometry PNGs.')
    parser.add_argument('-o', '--output', help='Output animation filename (.gif or .mp4).')
    parser.add_argument('--fps', type=float, default=5, help='Frames per second (default: 5).')
    parser.add_argument('--dpi', type=int, default=100, help='Output resolution (default: 100 dpi).')
    args = parser.parse_args()
    output = Path(args.output) if args.output else Path(args.filename).with_name(
        Path(args.filename).stem + '_animation.gif'
    )
    if not np.isfinite(args.fps) or args.fps <= 0 or args.dpi <= 0:
        parser.error('--fps and --dpi must be finite and positive.')
    if output.suffix.lower() == '.gif':
        writer = PillowWriter(fps=args.fps)
    elif output.suffix.lower() == '.mp4':
        if not writers.is_available('ffmpeg'):
            parser.error('MP4 output requires ffmpeg; use a .gif output instead.')
        writer = FFMpegWriter(fps=args.fps)
    else:
        parser.error('Output must have a .gif or .mp4 extension.')
    try:
        animation = create_animation(Path(args.filename), Path(args.plot_dir), args.fps)
    except (ValueError, FileNotFoundError) as error:
        parser.error(str(error))
    try:
        animation.save(output, writer=writer, dpi=args.dpi)
    finally:
        plt.close(animation._fig)
    print(f'Saved animation to: {output.resolve()}')


if __name__ == '__main__':
    main()
