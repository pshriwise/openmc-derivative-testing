"""Animate saved geometry images alongside the history of a parameter sweep."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter, writers
import numpy as np

from config import SURFACE_PARAMETER_NAME, SURFACE_PARAMETER_UNITS
from plot_parameter_sweep import load_sweep, finite_difference_derivative


def geometry_image_paths(plot_dir, parameter):
    """Match geometry files by run index and verify their saved parameter values."""
    paths = []
    for index, value in enumerate(parameter):
        matches = list(Path(plot_dir).glob(f'run_{index:04d}_parameter_*.png'))
        if len(matches) != 1:
            raise ValueError(
                f'Expected one geometry image for run {index} in {plot_dir}; '
                f'found {len(matches)}.'
            )
        path = matches[0]
        saved_parameter = float(path.stem.split('_parameter_', 1)[1])
        if not np.isclose(saved_parameter, value, rtol=1e-7, atol=1e-12):
            raise ValueError(
                f'{path} has parameter {saved_parameter}, but sweep run {index} '
                f'has parameter {value}.'
            )
        paths.append(path)
    return paths


def padded_limits(values):
    lower, upper = float(np.min(values)), float(np.max(values))
    padding = 0.05 * (upper - lower) if upper > lower else max(abs(lower) * 0.05, 1e-6)
    return lower - padding, upper + padding


def create_animation(input_path, plot_dir, fps=5):
    """Build frames in saved run order, using only observed points in each frame."""
    if fps <= 0:
        raise ValueError('fps must be positive.')
    results = load_sweep(input_path)
    if results.ndim != 1 or len(results) == 0:
        raise ValueError('Sweep results must be a nonempty one-dimensional array.')
    parameter = results['parameter']
    mean = results['tally_mean']
    estimate = results['geometry_derivative']
    if not np.all(np.isfinite([parameter, mean, estimate])):
        raise ValueError('Sweep parameter, tally mean, and derivative must be finite.')
    if len(np.unique(parameter)) != len(parameter):
        raise ValueError('Sweep parameter values must be unique.')
    image_paths = geometry_image_paths(plot_dir, parameter)

    # Each prefix is differentiated independently: later runs cannot affect
    # the finite-difference values displayed in an earlier frame.
    differences = [np.empty(0)] + [
        finite_difference_derivative(parameter[:count], mean[:count])
        for count in range(2, len(results) + 1)
    ]
    fig = plt.figure(figsize=(12, 7), layout='constrained')
    grid = fig.add_gridspec(3, 2, width_ratios=(1.4, 1))
    geometry_axis = fig.add_subplot(grid[:, 0])
    image = geometry_axis.imshow(plt.imread(image_paths[0]))
    geometry_axis.set_axis_off()
    mean_axis = fig.add_subplot(grid[0, 1])
    estimate_axis = fig.add_subplot(grid[1, 1], sharex=mean_axis)
    difference_axis = fig.add_subplot(grid[2, 1], sharex=mean_axis)
    mean_line, = mean_axis.plot([], [], 'o-', markersize=3, color='tab:blue')
    estimate_line, = estimate_axis.plot([], [], 'o-', markersize=3, color='tab:orange')
    difference_line, = difference_axis.plot([], [], 'o-', markersize=3, color='tab:green')
    pending = difference_axis.text(
        0.5, 0.5, 'Available after two runs', transform=difference_axis.transAxes,
        ha='center', va='center',
    )
    mean_axis.set_title('Tally mean')
    estimate_axis.set_title('Computed geometry derivative')
    difference_axis.set_title('Finite-difference derivative of observed tally values')
    mean_axis.set_ylabel('Tally mean')
    estimate_axis.set_ylabel('Derivative')
    difference_axis.set_ylabel('Derivative')
    difference_axis.set_xlabel(f'Iron sphere {SURFACE_PARAMETER_NAME} [{SURFACE_PARAMETER_UNITS}]')
    mean_axis.set_ylim(*padded_limits(mean))
    # Keep the derivative panels on the same fixed finite-difference scale.
    available = np.concatenate(differences)
    derivative_limits = padded_limits(available if available.size else estimate)
    for axis in (estimate_axis, difference_axis):
        axis.set_ylim(*derivative_limits)
    cursors = []
    for axis in (mean_axis, estimate_axis, difference_axis):
        axis.set_xlim(*padded_limits(parameter))
        axis.grid(True, alpha=0.3)
        cursors.append(axis.axvline(parameter[0], color='tab:red', linestyle='--', alpha=0.6))
    mean_axis.tick_params(labelbottom=False)
    estimate_axis.tick_params(labelbottom=False)
    title = fig.suptitle('')

    def update(index):
        count = index + 1
        image.set_data(plt.imread(image_paths[index]))
        mean_line.set_data(parameter[:count], mean[:count])
        estimate_line.set_data(parameter[:count], estimate[:count])
        difference_line.set_data(parameter[:count] if count >= 2 else [], differences[index])
        pending.set_visible(count < 2)
        for cursor in cursors:
            cursor.set_xdata([parameter[index], parameter[index]])
        title.set_text(f'Run {index} ({count}/{len(results)}) — {SURFACE_PARAMETER_NAME} = {parameter[index]:.8g} {SURFACE_PARAMETER_UNITS}')
        return image, mean_line, estimate_line, difference_line, pending, title, *cursors

    animation = FuncAnimation(
        fig, update, frames=len(results), interval=1000 / fps,
        blit=False, repeat=True, cache_frame_data=False,
    )
    return animation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('filename', nargs='?', default='parameter_sweep.npy')
    parser.add_argument('--plot-dir', default='geom_plots', help='Directory of run-indexed geometry PNGs.')
    parser.add_argument('-o', '--output', help='Output animation filename (.gif or .mp4).')
    parser.add_argument('--fps', type=float, default=5, help='Frames per second (default: 5).')
    parser.add_argument('--dpi', type=int, default=100, help='Output resolution (default: 100 dpi).')
    args = parser.parse_args()
    output = Path(args.output) if args.output else Path(args.filename).with_name(
        Path(args.filename).stem + '_animation.gif'
    )
    if args.fps <= 0 or args.dpi <= 0:
        parser.error('--fps and --dpi must be positive.')
    if output.suffix.lower() == '.gif':
        writer = PillowWriter(fps=args.fps)
    elif output.suffix.lower() == '.mp4':
        if not writers.is_available('ffmpeg'):
            parser.error('MP4 output requires ffmpeg; use a .gif output instead.')
        writer = FFMpegWriter(fps=args.fps)
    else:
        parser.error('Output must have a .gif or .mp4 extension.')
    try:
        animation = create_animation(Path(args.filename), Path(args.plot_dir), fps=args.fps)
    except (ValueError, FileNotFoundError) as error:
        parser.error(str(error))
    try:
        animation.save(output, writer=writer, dpi=args.dpi)
    finally:
        plt.close(animation._fig)
    print(f'Saved animation to: {output.resolve()}')


if __name__ == '__main__':
    main()
