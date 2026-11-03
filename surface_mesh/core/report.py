"""Objective histories and saved-image reports for one optimization run."""

import csv
import io
import json
from pathlib import Path

from .runtime import atomic_write


def accepted_evaluations(history):
    """Pair accepted-step indices with the latest completed estimate per mesh."""
    latest = {entry['mesh']: entry for entry in history['evaluations']
              if entry['status'] == 'complete'}
    return [(step, latest[mesh]) for step, mesh in enumerate(history['accepted_sequence'])
            if mesh in latest]


def draw_accepted_history(axis, history, current_step=None):
    """Plot accepted geometries, retaining refined estimates after resampling."""
    rows = accepted_evaluations(history)
    x = [step for step, _ in rows]
    y = [entry['tally_mean'] for _, entry in rows]
    errors = [entry['tally_std_dev'] for _, entry in rows]
    axis.errorbar(x, y, yerr=errors, fmt='o-', color='#b0bec5' if current_step is not None else '#176b9a',
                  markersize=3, linewidth=1.5, capsize=2)
    if current_step is not None:
        shown = [(step, entry) for step, entry in rows if step <= current_step]
        axis.plot([step for step, _ in shown], [entry['tally_mean'] for _, entry in shown],
                  'o-', color='#176b9a', markersize=3, zorder=3)
        selected = next((entry for step, entry in rows if step == current_step), None)
        if selected:
            axis.scatter([current_step], [selected['tally_mean']], s=65, color='#d65b28', zorder=4)
    axis.set_xlabel('Accepted update (0 = initial geometry)')
    axis.set_ylabel('Tally mean')
    axis.set_title('Accepted geometry • latest estimates, error bars ±1 standard deviation')
    axis.grid(alpha=.2)
    if x:
        axis.set_xlim(-.5, max(x[-1] + .5, .5))


def save_objective_history(root, history):
    """Save the accepted-step trajectory and every completed evaluation."""
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    completed = [entry for entry in history['evaluations'] if entry['status'] == 'complete']
    if not completed:
        return
    root = Path(root)
    accepted = set(history['accepted_sequence'])
    trial_status = {entry['mesh']: entry['status'] for entry in history.get('trials', []) if entry.get('mesh')}
    figure = Figure(figsize=(12, 8), dpi=150, layout='constrained')
    FigureCanvasAgg(figure)
    top, bottom = figure.subplots(2, 1)
    draw_accepted_history(top, history)
    bottom.plot([e['index'] for e in completed], [e['tally_mean'] for e in completed],
                color='#cfd8dc', linewidth=.9, zorder=0)
    for selected, label, color, marker in [
            (True, 'Initial / accepted geometry', '#176b9a', 'o'),
            (False, 'Unaccepted trial', '#d65b28', 'x')]:
        rows = [entry for entry in completed if (entry['mesh'] in accepted) == selected]
        if rows:
            bottom.errorbar([e['index'] for e in rows], [e['tally_mean'] for e in rows],
                            yerr=[e['tally_std_dev'] for e in rows], fmt=marker,
                            color=color, markersize=4, capsize=2, label=label)
    bottom.set_xlabel('Transport evaluation index (includes resampling and unaccepted trials)')
    bottom.set_ylabel('Tally mean')
    bottom.set_title('All completed evaluations • error bars ±1 standard deviation')
    bottom.grid(alpha=.2)
    bottom.legend()
    figure.suptitle(f'{root.name}\nOptimization objective history — {history["status"]}')
    for extension in ('png', 'svg'):
        atomic_write(root / f'objective_history.{extension}',
                     lambda stream, fmt=extension: figure.savefig(stream, format=fmt))
    figure.clear()
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(['evaluation', 'mesh', 'tally_mean', 'tally_std_dev', 'seed', 'budget',
                     'accepted_geometry', 'trial_status', 'particles_per_batch', 'num_realizations'])
    for entry in completed:
        writer.writerow([entry['index'], entry['mesh'], entry['tally_mean'], entry['tally_std_dev'],
                         entry.get('seed'), entry.get('budget'), entry['mesh'] in accepted,
                         trial_status.get(entry['mesh'], 'initial'), entry.get('particles'),
                         entry.get('num_realizations')])
    encoded = buffer.getvalue().encode()
    atomic_write(root / 'objective_history.csv', lambda stream: stream.write(encoded))


def regenerate(root):
    """Rebuild plots and GIFs from a stopped run without transport or rendering."""
    from .plots import update_accepted_animation

    root = Path(root).resolve()
    history = json.loads((root / 'history.json').read_text())
    if history['status'] == 'running':
        raise ValueError('Regeneration requires a stopped run; live runs update their own reports')
    save_objective_history(root, history)
    update_accepted_animation(root, history)
