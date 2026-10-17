import argparse
from pathlib import Path

import numpy as np
import openmc
import openmc.lib

from geometry_plots import parameter_plot_extent, render_current_parameter

DEFAULT_OUTPUT = 'collision_parameter_sweep.npz'
DEFAULT_NUM_VALUES = 100
DEFAULT_MIN_COLLISION = 1
DEFAULT_MAX_COLLISION = 20
DEFAULT_SEED = 10
DEFAULT_SCORE = 'flux'


def get_constraints(model):
    """Return the lower and upper bounds for the swept geometry parameter."""
    water_sphere = model.geometry.get_all_surfaces()[2]
    return 0.01, water_sphere.x0 - water_sphere.r


def update_parameter(model, new_radius):
    """Update the swept geometry parameter, the radius of surface 1."""
    iron_sphere = model.geometry.get_all_surfaces()[1]
    iron_sphere.r = new_radius


def disable_tally_triggers(model):
    """Disable tally triggers on the model and all tallies."""
    model.settings.trigger_active = False
    for tally in model.tallies:
        tally.triggers = []


def next_filter_id(model):
    """Return an unused filter ID for the model."""
    filter_ids = [
        tally_filter.id
        for tally in model.tallies
        for tally_filter in tally.filters
        if tally_filter.id is not None
    ]
    return max(filter_ids, default=0) + 1


def add_collision_filter_to_all_tallies(model, collision_bins):
    """Append a collision filter to all tallies, replacing any existing one."""
    collision_filter = openmc.CollisionFilter(
        collision_bins,
        filter_id=next_filter_id(model),
    )

    for tally in model.tallies:
        tally.filters = [
            tally_filter
            for tally_filter in tally.filters
            if not isinstance(tally_filter, openmc.CollisionFilter)
        ]
        tally.filters.append(collision_filter)


def prepare_model(model_xml, radius, collision_bins, disable_triggers):
    """Load and configure a model for one collision-filtered sweep point."""
    model = openmc.Model.from_model_xml(model_xml)
    model.geometry_derivatives = None
    update_parameter(model, radius)
    add_collision_filter_to_all_tallies(model, collision_bins)

    if disable_triggers:
        disable_tally_triggers(model)

    return model


def score_column_indices(tally, score_name):
    """Return flattened tally result columns matching a score name."""
    scores = tally.scores
    if score_name not in scores:
        return []

    score_index = scores.index(score_name)
    num_scores = len(scores)
    num_columns = tally.mean.shape[1]
    num_nuclides = num_columns // num_scores
    return [nuclide * num_scores + score_index for nuclide in range(num_nuclides)]


def collapse_to_collision_bins(tally, collision_bins, score_name):
    """Return mean and std. dev. arrays summed over non-collision bins."""
    score_columns = score_column_indices(tally, score_name)
    if not score_columns:
        count = len(collision_bins)
        return np.full(count, np.nan), np.full(count, np.nan)

    mean = tally.mean[:, score_columns].sum(axis=1)
    std_dev = np.sqrt((tally.std_dev[:, score_columns]**2).sum(axis=1))

    num_collision_bins = len(collision_bins)
    if mean.size % num_collision_bins != 0:
        raise ValueError(
            f'Tally {tally.id} has {mean.size} filter bins, which is not '
            f'divisible by {num_collision_bins} collision bins.'
        )

    other_filter_bins = mean.size // num_collision_bins
    mean = mean.reshape(other_filter_bins, num_collision_bins).sum(axis=0)
    std_dev = np.sqrt(
        (std_dev.reshape(other_filter_bins, num_collision_bins)**2).sum(axis=0)
    )
    return mean, std_dev


def run_collision_filtered_model(model, collision_bins, score_name, runtime_args,
                                 run_index=0, plot_extent=None):
    """Run one prepared model in a temporary OpenMC session."""
    tally_ids = np.array([tally.id for tally in model.tallies], dtype=int)
    tally_names = np.array([tally.name or f'tally {tally.id}' for tally in model.tallies])

    mean = np.empty((len(tally_ids), len(collision_bins)))
    std_dev = np.empty_like(mean)

    if plot_extent is None:
        lower_bound, upper_bound = get_constraints(model)
        plot_extent = parameter_plot_extent(
            model, upper_bound, parameter_name='r', min_parameter=lower_bound,
        )
    with openmc.lib.TemporarySession(model, args=runtime_args, output=False):
        image = render_current_parameter(
            model, run_index, plot_extent, parameter_name='r',
        )
        openmc.lib.run(output=False)
        for index, tally_id in enumerate(tally_ids):
            tally = openmc.lib.tallies[int(tally_id)]
            mean[index], std_dev[index] = collapse_to_collision_bins(
                tally,
                collision_bins,
                score_name,
            )

    return tally_ids, tally_names, mean, std_dev, image


def collect_collision_parameter_sweep(
    model_xml='model.xml',
    num_values=DEFAULT_NUM_VALUES,
    collision_bins=None,
    output_path=DEFAULT_OUTPUT,
    score_name=DEFAULT_SCORE,
    seed=DEFAULT_SEED,
    disable_triggers=False,
):
    """Run a collision-filtered parameter sweep and save named arrays."""
    model_xml = Path(model_xml)
    if collision_bins is None:
        collision_bins = np.arange(DEFAULT_MIN_COLLISION, DEFAULT_MAX_COLLISION + 1)
    collision_bins = np.asarray(collision_bins, dtype=int)

    base_model = openmc.Model.from_model_xml(model_xml)
    lower_bound, upper_bound = get_constraints(base_model)
    radii = np.linspace(lower_bound, upper_bound, num_values, endpoint=True)
    plot_extent = parameter_plot_extent(base_model, upper_bound, parameter_name='r',
                                       min_parameter=lower_bound)
    runtime_args = ['-s', str(seed)]

    tally_ids = None
    tally_names = None
    means = []
    std_devs = []
    actual_radii = []
    images = []

    for index, radius in enumerate(radii, start=1):
        model = prepare_model(model_xml, float(radius), collision_bins, disable_triggers)
        actual_radius = model.geometry.get_all_surfaces()[1].r
        actual_radii.append(actual_radius)
        print(
            f'Running radius {index}/{len(radii)}: '
            f'requested={radius}, actual={actual_radius}'
        )
        run_tally_ids, run_tally_names, mean, std_dev, image = run_collision_filtered_model(
            model,
            collision_bins,
            score_name,
            runtime_args,
            run_index=index - 1,
            plot_extent=plot_extent,
        )

        if tally_ids is None:
            tally_ids = run_tally_ids
            tally_names = run_tally_names
        elif not np.array_equal(tally_ids, run_tally_ids):
            raise RuntimeError('Tally IDs changed during the sweep.')

        images.append(image)
        means.append(mean)
        std_devs.append(std_dev)

    mean = np.stack(means, axis=1)
    std_dev = np.stack(std_devs, axis=1)
    collision_sum = np.nansum(mean, axis=2)
    actual_radii = np.asarray(actual_radii)
    images = np.stack(images)

    np.savez_compressed(
        output_path,
        image=images,
        radii=radii,
        actual_radii=actual_radii,
        collision_bins=collision_bins,
        tally_ids=tally_ids,
        tally_names=tally_names,
        mean=mean,
        std_dev=std_dev,
        collision_sum=collision_sum,
        score_name=np.array(score_name),
        model_xml=np.array(str(model_xml)),
    )
    print(f'Saved collision sweep results to: {output_path}')

    return {
        'image': images,
        'radii': radii,
        'actual_radii': actual_radii,
        'collision_bins': collision_bins,
        'tally_ids': tally_ids,
        'tally_names': tally_names,
        'mean': mean,
        'std_dev': std_dev,
        'collision_sum': collision_sum,
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description='Run a collision-filtered parameter sweep from model XML.'
    )
    parser.add_argument(
        'model_xml',
        nargs='?',
        default='model.xml',
        help='Path to the source model XML file.',
    )
    parser.add_argument(
        '-o',
        '--output',
        default=DEFAULT_OUTPUT,
        help='Path where the sweep results should be saved.',
    )
    parser.add_argument(
        '-n',
        '--num-values',
        type=int,
        default=DEFAULT_NUM_VALUES,
        help='Number of evenly spaced radius values to evaluate.',
    )
    parser.add_argument(
        '--min-collision',
        type=int,
        default=DEFAULT_MIN_COLLISION,
        help='Minimum collision count included in the collision filter.',
    )
    parser.add_argument(
        '--max-collision',
        type=int,
        default=DEFAULT_MAX_COLLISION,
        help='Maximum collision count included in the collision filter.',
    )
    parser.add_argument(
        '--score',
        default=DEFAULT_SCORE,
        help='Tally score to collect and plot.',
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=DEFAULT_SEED,
        help='OpenMC random-number seed passed with -s.',
    )
    parser.add_argument(
        '--disable-triggers',
        action='store_true',
        help='Disable tally triggers after loading the source model.',
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.min_collision > args.max_collision:
        raise ValueError('--min-collision must be <= --max-collision.')

    collision_bins = np.arange(args.min_collision, args.max_collision + 1)
    collect_collision_parameter_sweep(
        model_xml=args.model_xml,
        num_values=args.num_values,
        collision_bins=collision_bins,
        output_path=args.output,
        score_name=args.score,
        seed=args.seed,
        disable_triggers=args.disable_triggers,
    )


if __name__ == '__main__':
    main()
