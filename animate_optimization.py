"""Animate optimization geometry and tally/derivative histories by run number."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter, writers
from matplotlib.ticker import MaxNLocator
import numpy as np

from animate_parameter_sweep import geometry_image_paths, padded_limits
from config import SURFACE_PARAMETER_NAME, SURFACE_PARAMETER_UNITS
from plot_parameter_sweep import load_sweep


def create_animation(input_path, plot_dir, fps=5):
    """Preserve evaluation order, including repeated parameter values and line-search runs."""
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError('fps must be finite and positive.')
    results = load_sweep(input_path)
    if results.ndim != 1 or len(results) == 0:
        raise ValueError('Optimization history must be a nonempty one-dimensional array.')
    parameter = results['parameter']
    mean = results['tally_mean']
    derivative = results['geometry_derivative']
    if not np.all(np.isfinite([parameter, mean, derivative])):
        raise ValueError('Radius, tally mean, and derivative must be finite.')
    image_paths = geometry_image_paths(plot_dir, parameter)
    runs = np.arange(len(results))

    fig = plt.figure(figsize=(13, 7), layout='constrained')
    grid = fig.add_gridspec(2, 2, width_ratios=(1.3, 1))
    geometry_axis = fig.add_subplot(grid[:, 0])
    image = geometry_axis.imshow(plt.imread(image_paths[0]))
    geometry_axis.set_axis_off()
    mean_axis = fig.add_subplot(grid[0, 1])
    derivative_axis = fig.add_subplot(grid[1, 1], sharex=mean_axis)
    mean_line, = mean_axis.plot([], [], 'o-', markersize=4, color='tab:blue')
    derivative_line, = derivative_axis.plot([], [], 'o-', markersize=4, color='tab:orange')
    mean_axis.set_title(f'Tally mean (point labels: {SURFACE_PARAMETER_NAME} in {SURFACE_PARAMETER_UNITS})')
    derivative_axis.set_title(f'Geometry derivative (point labels: {SURFACE_PARAMETER_NAME} in {SURFACE_PARAMETER_UNITS})')
    mean_axis.set_ylabel('Tally mean')
    derivative_axis.set_ylabel('Derivative')
    derivative_axis.set_xlabel('Evaluation / run number')
    mean_axis.tick_params(labelbottom=False)
    cursors = []
    labels = []
    for axis, values in ((mean_axis, mean), (derivative_axis, derivative)):
        axis.set_xlim(-0.5, max(len(results) - 0.5, 0.5))
        axis.set_ylim(*padded_limits(values))
        axis.xaxis.set_major_locator(MaxNLocator(integer=True))
        axis.grid(True, alpha=0.3)
        cursors.append(axis.axvline(0, color='tab:red', linestyle='--', alpha=0.6))
        for index, (value, parameter_value) in enumerate(zip(values, parameter)):
            above = index % 2 == 0
            labels.append(axis.annotate(
                f'{SURFACE_PARAMETER_NAME}={parameter_value:.4g}', xy=(index, value),
                xytext=(0, 8 if above else -10), textcoords='offset points',
                ha='center', va='bottom' if above else 'top', fontsize=7,
                visible=False,
            ))
    title = fig.suptitle('')

    def update(index):
        count = index + 1
        image.set_data(plt.imread(image_paths[index]))
        mean_line.set_data(runs[:count], mean[:count])
        derivative_line.set_data(runs[:count], derivative[:count])
        for label_index, label in enumerate(labels):
            label.set_visible(label_index % len(results) < count)
        for cursor in cursors:
            cursor.set_xdata([index, index])
        title.set_text(
            f'Optimization run {index} ({count}/{len(results)}) '
            f'— {SURFACE_PARAMETER_NAME} = {parameter[index]:.8g} {SURFACE_PARAMETER_UNITS}'
        )
        return image, mean_line, derivative_line, title, *cursors, *labels

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
