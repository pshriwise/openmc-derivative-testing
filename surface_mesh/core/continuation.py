"""Continue an INT run from its last accepted mesh with the optimizer state it had."""

from dataclasses import dataclass, replace
import json
from pathlib import Path

import numpy as np

from surface_mesh.mesh import load_mesh
from surface_mesh.int_optimizer import optimize_vertices
from surface_mesh.int_optimizer.config import load_config

from .settings import load_run_config
from .sampling import AdaptiveSampler

# Replayed Adam steps run in float32; saved candidates were written from them.
REPLAY_TOLERANCE_CM = 1e-5


@dataclass
class Continuation:
    """Everything a continuation needs to take the source run's next update."""
    source: Path
    mesh: Path
    int_config: object
    settings: object
    next_iteration: int
    volume_target: float
    learning_rate: float
    adam_state: dict | None
    particles: int
    seed: int
    observations: list
    decisions: list
    provenance: dict

    def sampler(self, config):
        """Rebuild the adaptive sampler as it stood before the next decision."""
        sampler = AdaptiveSampler(config)
        sampler.observations = [dict(row) for row in self.observations]
        if sampler.observations:
            sampler.baseline_gradient = sampler.observations[0]['gradient_norm']
        increases = [d['iteration'] for d in self.decisions if d['increase']]
        sampler.last_increase = increases[-1] if increases else None
        sampler.streak = self.decisions[-1].get('streak', 0) if self.decisions else 0
        return sampler


def load_adam(path):
    with np.load(path) as arrays:
        return dict(vertices=arrays['vertices'], m=arrays['m'], v=arrays['v'], t=int(arrays['t']))


def replay_adam(source, trials, config, volume_target):
    """Recompute Adam's state by repeating each accepted single-step update.

    Each replayed step must reproduce the saved candidate coordinates; any
    larger difference means the state cannot be trusted and nothing resumes.
    """
    if config.iterations != 1:
        raise ValueError('Adam replay requires one Adam step per update; save states instead')
    state, worst = None, 0.
    for trial in trials:
        if trial.get('adam_state'):
            state = load_adam(source / trial['adam_state'])
        else:
            parent = load_mesh(source / trial['parent_mesh'], validate_geometry=False)
            with np.load((source / trial.get('optimization_mesh', trial['mesh'])).with_suffix('.npz')) as arrays:
                gradient, expected = arrays['gradient'], arrays['coordinates']
            step = replace(config, learning_rate=trial.get('learning_rate', config.learning_rate))
            result = optimize_vertices(parent.coordinates, parent.faces, gradient, step,
                                       volume_target=volume_target, adam_state=state)
            error = float(np.abs(result.vertices - expected).max())
            if error > REPLAY_TOLERANCE_CM:
                raise ValueError(f"Replay of update {trial['iteration']} differs from its saved mesh "
                                 f'by {error:.3g} cm; refusing to continue')
            worst = max(worst, error)
            state = result.adam_state
        if trial.get('remeshing', {}).get('adam_state') == 'reset':
            state = None
    return state, worst


def continuation_state(source, int_config=None, settings=None):
    """Read a stopped or running INT run and return its last accepted state.

    An update still in progress, or one that never completed transport, is
    ignored. Seeds continue the source run's sequence so no history is reused.
    """
    source = Path(source).resolve()
    history = json.loads((source / 'history.json').read_text())
    if history.get('mode') != 'int_inspired':
        raise ValueError(f'Not an INT run: {source}')
    provenance = json.loads((source / 'provenance.json').read_text())
    used = load_config(source / 'int_inspired_config.json')
    settings = settings or load_run_config(source / 'run_settings.json')
    int_config = int_config or used
    accepted = [t for t in history['trials'] if t['status'] == 'accepted']
    last = accepted[-1] if accepted else None
    mesh = source / (last['mesh'] if last else history['initial_mesh'])
    if last and history['accepted_sequence'][-1] != last['mesh']:
        raise ValueError('Accepted sequence and trials disagree; cannot identify the last mesh')
    volume_target = provenance['volume_target']
    if not used.persist_adam_state or not accepted:
        state, method, error = None, 'fresh', None
    elif last.get('adam_state'):
        state, method, error = load_adam(source / last['adam_state']), 'saved', None
        if last.get('remeshing', {}).get('adam_state') == 'reset':
            state = None
    else:
        state, error = replay_adam(source, accepted, used, volume_target)
        method = 'replayed'
    learning_rate = used.learning_rate
    if last:
        learning_rate = last.get('remeshing', {}).get('next_learning_rate',
                                                      last.get('learning_rate', learning_rate))
    iteration = last['iteration'] if last else 0
    evaluation = (history['evaluations'][last['candidate_evaluation']] if last
                  else history['evaluations'][0])
    particles = evaluation['particles'] // evaluation['budget']
    # Decisions before update k observe update k - 1; the pending one is redone.
    sampling = history.get('adaptive_sampling', {})
    observations = [o for o in sampling.get('observations', []) if o['iteration'] <= iteration]
    decisions = [d for d in sampling.get('decisions', []) if d['iteration'] < iteration]
    seed = settings.seed + 104729 * len(history['evaluations'])
    record = dict(source=str(source), mesh=str(mesh), last_accepted_iteration=iteration,
                  source_status=history['status'], source_evaluations=len(history['evaluations']),
                  adam_state=method, adam_steps=None if state is None else state['t'],
                  replay_max_error_cm=error, learning_rate=learning_rate, particles=particles,
                  seed=seed, volume_target=volume_target,
                  source_initial_tally_mean=history['evaluations'][0].get('tally_mean'),
                  source_model_sha256=provenance['model_sha256'],
                  source_library_sha256=provenance['library_sha256'])
    return Continuation(source, mesh, int_config, settings, iteration + 1, volume_target, learning_rate,
                        state, particles, seed, observations, decisions, record)
