import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REQUIRED_FIELDS = (
    'parameter',
    'tally_mean',
    'tally_std_dev',
    'geometry_derivative',
    'dJ',
    'datt',
    'parameter_names',
    'case_metadata',
)

def load_sweep(path):
    """Load parameter sweep results and validate expected named fields."""
    results = np.load(path)
    if results.dtype.names is None:
        raise ValueError(f'{path} does not contain a structured NumPy array.')

    missing_fields = set(REQUIRED_FIELDS) - set(results.dtype.names)
    if missing_fields:
        missing = ', '.join(sorted(missing_fields))
        raise ValueError(f'{path} is missing required field(s): {missing}')

    return results


def history_metadata(results):
    """Read the case settings embedded in a nonempty history."""
    if results.ndim != 1 or len(results) == 0:
        raise ValueError('History must be a nonempty one-dimensional array.')
    encoded = results['case_metadata']
    if not np.all(encoded == encoded[0]):
        raise ValueError('Case metadata must be consistent across history rows.')
    return json.loads(str(encoded[0]))


def finite_difference_derivative(parameter, values):
    """Estimate d(values)/d(parameter) from the evaluated sweep results."""
    if len(parameter) < 2:
        raise ValueError('At least two sweep points are needed for derivatives.')
    if len(np.unique(parameter)) != len(parameter):
        raise ValueError('Sweep parameter values must be unique.')

    edge_order = 2 if len(parameter) > 2 else 1
    return np.gradient(values, parameter, edge_order=edge_order)


def plot_parameter_sweep(input_path, output_path, simple=False):
    """Plot sweep result fields against the swept parameter parameter."""
    results = load_sweep(input_path)
    metadata = history_metadata(results)
    parameter_label = f"Iron sphere {metadata['parameter']} [{metadata['units']}]"
    score = metadata['tally']['score']
    tally_label = f'{score} tally mean'
    order = np.argsort(results['parameter'])
    parameter = results['parameter'][order]
    tally_mean = results['tally_mean'][order]
    tally_std_dev = results['tally_std_dev'][order]
    geometry_derivative = results['geometry_derivative'][order]
    dJ = results['dJ'][order]
    datt = results['datt'][order]
    component_sum = dJ + datt
    finite_difference = finite_difference_derivative(parameter, tally_mean)
    derivative_difference = geometry_derivative - finite_difference
    finite_difference_component_ratio = np.divide(
        finite_difference - dJ,
        datt,
        out=np.full_like(finite_difference, np.nan),
        where=datt != 0.0,
    )

    min_index = np.argmin(tally_mean)
    min_parameter = parameter[min_index]
    min_parameter_label = f'min mean parameter = {min_parameter:.6g}'

    fig = plt.figure(figsize=(11, 6) if simple else (11, 12), constrained_layout=True)
    grid = fig.add_gridspec(2 if simple else 4, 2)
    mean_axis = fig.add_subplot(grid[0, :])
    derivative_axis = fig.add_subplot(grid[1, 0], sharex=mean_axis)
    finite_difference_axis = fig.add_subplot(grid[1, 1], sharex=mean_axis)
    if not simple:
        difference_axis = fig.add_subplot(grid[2, 0], sharex=mean_axis)
        component_axis = fig.add_subplot(grid[2, 1], sharex=mean_axis)
        ratio_axis = fig.add_subplot(grid[3, :], sharex=mean_axis)

    mean_axis.fill_between(
        parameter,
        tally_mean - 3.0 * tally_std_dev,
        tally_mean + 3.0 * tally_std_dev,
        color='tab:blue',
        alpha=0.2,
        label='Mean +/- 3 std. dev.',
    )
    mean_axis.plot(
        parameter,
        tally_mean,
        marker='o',
        markersize=3,
        linewidth=1,
        color='tab:blue',
        label='Tally mean',
    )
    mean_axis.axvline(
        min_parameter,
        color='tab:red',
        linestyle='--',
        linewidth=1,
        label=min_parameter_label,
    )
    mean_axis.set_title(f'{tally_label} with +/- 3 std. dev.')
    mean_axis.set_ylabel('Tally Mean')
    mean_axis.grid(True, alpha=0.3)
    mean_axis.legend(loc='best')
    mean_axis.tick_params(labelbottom=False)

    derivative_axis.plot(
        parameter,
        geometry_derivative,
        marker='o',
        markersize=3,
        linewidth=1,
    )
    derivative_axis.axvline(
        min_parameter,
        color='tab:red',
        linestyle='--',
        linewidth=1,
        label=min_parameter_label,
    )
    derivative_axis.set_title('Tallied geometry derivative')
    derivative_axis.set_ylabel('Geometry Derivative')
    derivative_axis.grid(True, alpha=0.3)
    derivative_axis.legend(loc='best')
    derivative_axis.tick_params(labelbottom=False)

    finite_difference_axis.plot(
        parameter,
        finite_difference,
        marker='o',
        markersize=3,
        linewidth=1,
    )
    finite_difference_axis.axvline(
        min_parameter,
        color='tab:red',
        linestyle='--',
        linewidth=1,
        label=min_parameter_label,
    )
    finite_difference_axis.set_title('Finite-difference derivative')
    finite_difference_axis.set_ylabel('Finite Difference')
    finite_difference_axis.grid(True, alpha=0.3)
    finite_difference_axis.legend(loc='best')
    finite_difference_axis.tick_params(labelbottom=False)

    derivative_limits = (np.min(finite_difference), np.max(finite_difference))
    derivative_axis.set_ylim(*derivative_limits)
    finite_difference_axis.set_ylim(*derivative_limits)

    if simple:
        for axis in (derivative_axis, finite_difference_axis):
            axis.set_xlabel('Parameter value')
            axis.tick_params(labelbottom=True)
    else:
        difference_axis.axhline(0.0, color='black', linewidth=1, alpha=0.5)
        difference_axis.plot(
            parameter,
            derivative_difference,
            marker='o',
            markersize=3,
            linewidth=1,
        )
        difference_axis.axvline(
            min_parameter,
            color='tab:red',
            linestyle='--',
            linewidth=1,
            label=min_parameter_label,
        )
        difference_axis.set_title('Tallied minus finite-difference derivative')
        difference_axis.set_xlabel('Parameter value')
        difference_axis.set_ylabel('Derivative Difference')
        difference_axis.grid(True, alpha=0.3)
        difference_axis.legend(loc='best')

        component_axis.axhline(0.0, color='black', linewidth=1, alpha=0.5)
        component_axis.plot(
            parameter,
            dJ,
            marker='o',
            markersize=3,
            linewidth=1,
            label='dJ',
        )
        component_axis.plot(
            parameter,
            datt,
            marker='o',
            markersize=3,
            linewidth=1,
            label='datt',
        )
        component_axis.plot(
            parameter,
            component_sum,
            marker='o',
            markersize=3,
            linewidth=1,
            label='dJ + datt',
        )
        component_axis.axvline(
            min_parameter,
            color='tab:red',
            linestyle='--',
            linewidth=1,
            label=min_parameter_label,
        )
        component_axis.set_title('Geometry derivative components')
        component_axis.set_xlabel('Parameter value')
        component_axis.set_ylabel('Component Value')
        component_axis.grid(True, alpha=0.3)
        component_axis.legend(loc='best')

        ratio_axis.axhline(1.0, color='black', linewidth=1, alpha=0.5)
        ratio_axis.plot(
            parameter,
            finite_difference_component_ratio,
            marker='o',
            markersize=3,
            linewidth=1,
        )
        ratio_axis.axvline(
            min_parameter,
            color='tab:red',
            linestyle='--',
            linewidth=1,
            label=min_parameter_label,
        )
        ratio_axis.set_title('(Finite-difference derivative - dJ) / datt')
        ratio_axis.set_xlabel('Parameter value')
        ratio_axis.set_ylabel('Ratio')
        ratio_axis.grid(True, alpha=0.3)
        ratio_axis.legend(loc='best')

    for axis in fig.axes:
        if axis.get_xlabel():
            axis.set_xlabel(parameter_label)
    case_name = metadata.get('name')
    fig.suptitle(f'Sweep: {case_name}' if case_name else 'Parameter Sweep Results')
    fig.savefig(output_path, dpi=200)
    return fig


def default_output_path(input_path):
    """Return the default plot filename for a sweep result filename."""
    return input_path.with_suffix('.png')


def main():
    parser = argparse.ArgumentParser(
        description='Plot OpenMC parameter sweep results against parameter.'
    )
    parser.add_argument(
        'filename',
        nargs='?',
        default='parameter_sweep.npy',
        help='Path to the structured NumPy sweep file.',
    )
    parser.add_argument(
        '-o',
        '--output',
        help='Path where the plot image should be written.',
    )
    parser.add_argument(
        '--simple',
        action='store_true',
        help='Show only the tally mean and the two derivative plots.',
    )
    parser.add_argument(
        '--show',
        action='store_true',
        help='Display the plot window after saving.',
    )
    args = parser.parse_args()

    input_path = Path(args.filename)
    if args.output is None:
        output_path = default_output_path(input_path)
    else:
        output_path = Path(args.output)

    plot_parameter_sweep(input_path, output_path, simple=args.simple)
    print(f'Saved plot to: {output_path}')

    if args.show:
        plt.show()


if __name__ == '__main__':
    main()
