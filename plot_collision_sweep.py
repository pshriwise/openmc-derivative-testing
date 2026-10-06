import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


REQUIRED_ARRAYS = (
    'radii',
    'collision_bins',
    'tally_ids',
    'tally_names',
    'mean',
    'std_dev',
    'collision_sum',
    'score_name',
)


def load_collision_sweep(path):
    """Load a collision-filtered parameter sweep result file."""
    data = np.load(path)
    missing = set(REQUIRED_ARRAYS) - set(data.files)
    if missing:
        missing_names = ', '.join(sorted(missing))
        raise ValueError(f'{path} is missing required array(s): {missing_names}')
    return data


def default_output_path(input_path):
    """Return the default plot filename for a sweep result filename."""
    return input_path.with_suffix('.png')


def select_tally_index(data, tally_id=None, tally_name=None):
    """Return the index of the requested tally in a sweep result file."""
    tally_ids = data['tally_ids']
    tally_names = data['tally_names'].astype(str)

    if tally_id is not None:
        matches = np.flatnonzero(tally_ids == tally_id)
        if len(matches) == 0:
            raise ValueError(f'Tally ID {tally_id} was not found.')
        return int(matches[0])

    if tally_name is not None:
        matches = np.flatnonzero(tally_names == tally_name)
        if len(matches) == 0:
            raise ValueError(f'Tally name {tally_name!r} was not found.')
        return int(matches[0])

    return 0


def label_trend(axis, x_values, y_values, label, color, y_offset=0):
    """Place a label next to the last finite point on a trend line."""
    finite = np.isfinite(x_values) & np.isfinite(y_values)
    if not np.any(finite):
        return

    x = x_values[finite][-1]
    y = y_values[finite][-1]
    axis.annotate(
        label,
        xy=(x, y),
        xytext=(8, y_offset),
        textcoords='offset points',
        color=color,
        fontsize='small',
        va='center',
        clip_on=False,
    )


def plot_collision_sweep(
    input_path,
    output_path,
    tally_id=None,
    tally_name=None,
    sigma=3.0,
):
    """Plot collision-filtered flux trends against the swept radius."""
    data = load_collision_sweep(input_path)
    tally_index = select_tally_index(data, tally_id=tally_id, tally_name=tally_name)

    if 'actual_radii' in data.files:
        radii = data['actual_radii']
    else:
        radii = data['radii']
    collision_bins = data['collision_bins']
    mean = data['mean'][tally_index]
    std_dev = data['std_dev'][tally_index]
    collision_sum = data['collision_sum'][tally_index]
    selected_tally_id = int(data['tally_ids'][tally_index])
    selected_tally_name = str(data['tally_names'][tally_index])
    score_name = str(data['score_name'])

    fig, axis = plt.subplots(figsize=(12, 7), constrained_layout=True)
    colors = plt.cm.viridis(np.linspace(0.0, 1.0, len(collision_bins)))

    label_offsets = np.linspace(
        -0.4 * len(collision_bins),
        0.4 * len(collision_bins),
        len(collision_bins),
    )
    for index, (collision_bin, color, label_offset) in enumerate(
        zip(collision_bins, colors, label_offsets)
    ):
        trend = mean[:, index]
        trend_std_dev = std_dev[:, index]
        axis.fill_between(
            radii,
            trend - sigma * trend_std_dev,
            trend + sigma * trend_std_dev,
            color=color,
            alpha=0.12,
            linewidth=0,
        )
        axis.plot(
            radii,
            trend,
            color=color,
            linewidth=1,
            marker='o',
            markersize=3,
            label=f'collision {collision_bin}',
        )
        label_trend(
            axis,
            radii,
            trend,
            f'c={collision_bin}',
            color,
            y_offset=label_offset,
        )

    axis.plot(
        radii,
        collision_sum,
        color='black',
        linewidth=2,
        linestyle='--',
        label='sum over collisions',
    )
    label_trend(axis, radii, collision_sum, 'sum', 'black')

    axis.set_title(
        f'{score_name} by collision count for {selected_tally_name} '
        f'(tally {selected_tally_id})'
    )
    axis.set_xlabel('Iron sphere radius')
    axis.set_ylabel(score_name)
    axis.grid(True, alpha=0.3)
    axis.margins(x=0.12)
    axis.legend(
        loc='center left',
        bbox_to_anchor=(1.02, 0.5),
        fontsize='small',
        title='Trends',
    )

    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    return fig


def parse_args():
    parser = argparse.ArgumentParser(
        description='Plot collision-filtered sweep results.'
    )
    parser.add_argument(
        'filename',
        nargs='?',
        default='collision_parameter_sweep.npz',
        help='Path to the collision sweep .npz file.',
    )
    parser.add_argument(
        '-o',
        '--output',
        help='Path where the plot image should be written.',
    )
    parser.add_argument(
        '--tally-id',
        type=int,
        help='Tally ID to plot. Defaults to the first tally in the file.',
    )
    parser.add_argument(
        '--tally-name',
        help='Tally name to plot. Defaults to the first tally in the file.',
    )
    parser.add_argument(
        '--sigma',
        type=float,
        default=3.0,
        help='Number of standard deviations used for shaded bands.',
    )
    parser.add_argument(
        '--show',
        action='store_true',
        help='Display the plot window after saving.',
    )
    return parser.parse_args()


def main():
    args = parse_args()
    input_path = Path(args.filename)
    if args.output is None:
        output_path = default_output_path(input_path)
    else:
        output_path = Path(args.output)

    plot_collision_sweep(
        input_path=input_path,
        output_path=output_path,
        tally_id=args.tally_id,
        tally_name=args.tally_name,
        sigma=args.sigma,
    )
    print(f'Saved plot to: {output_path}')

    if args.show:
        plt.show()


if __name__ == '__main__':
    main()
