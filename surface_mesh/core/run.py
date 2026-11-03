"""Fresh OpenMC/INT runs using the framework's existing visualization services."""

from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from surface_mesh.mesh import DeformationOptions, MeshConstraints, load_mesh
from surface_mesh.int_optimizer import IntConfig, optimize_vertices
from surface_mesh.int_optimizer.interchange import load_derivatives, write_displaced_h5m
from surface_mesh.int_optimizer.remesh import remesh_h5m
from surface_mesh.mesh.geometry import signed_volume
from surface_mesh.mesh.reporting import file_hash, write_report

from .evaluate import evaluate, validate_model_container
from .history import History
from .settings import IntRunSettings
from .models import build_model, model_source_path, objective_tally, source_position_in_mesh
from .progress import phase
from .mesh_motion import render_motion
from .sampling import AdaptiveSampler

DEFAULT_INPUT = Path(__file__).resolve().parents[1] / 'sphere.h5m'


def evaluate_int(mesh_path, directory, settings, seed, budget=1):
    """Use the shared evaluator with independent INT settings in its subprocess."""
    return evaluate(mesh_path, directory, settings, seed, budget, settings_kind='int')


def validate_domain(mesh, candidate, settings):
    """Check actual model containment and source clearance without shaping a step."""
    if settings.model_variant == 'box':
        lower = np.asarray(settings.box_container_lower) + settings.container_margin
        upper = np.asarray(settings.box_container_upper) - settings.container_margin
        outside = np.any(candidate <= lower) or np.any(candidate >= upper)
    else:
        outside = np.any(np.linalg.norm(candidate, axis=1) >=
                         settings.container_radius - settings.container_margin)
    if outside:
        raise ValueError('INT candidate leaves the CSG mesh container')
    # This supplies only the existing source-path check, not legacy projection.
    guard = MeshConstraints(mesh, options=DeformationOptions(check_self_intersections=False),
                            vertex_exclusion_center=source_position_in_mesh(settings),
                            vertex_exclusion_radius=settings.source_clearance)
    guard.validate_vertex_clearance(candidate)


def run(input_mesh, output, int_config, settings=None, *, evaluator=evaluate_int, continuation=None,
        allow_changed_model=False):
    """Evaluate a fresh baseline and refresh transport after each INT update.

    Accepted means a completed INT update with a completed transport
    evaluation. Measured improvement is reported, not an acceptance criterion.
    No legacy deformation, hard volume projection, or centroid fixing is used.

    A ``continuation`` (see ``int_continuation.continuation_state``) resumes
    the source run's numbering, volume target, learning rate, Adam state,
    particle count, sampling history, and seed sequence. Its baseline is a
    fresh evaluation of the last accepted mesh; ``max_iterations`` is the
    final update number, not a count of additional updates.
    """
    import openmc
    import openmc.lib
    from lxml import etree

    cfg = settings or IntRunSettings()
    if not isinstance(cfg, IntRunSettings) or not isinstance(int_config, IntConfig):
        raise TypeError('INT runs require IntRunSettings and IntConfig')
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(f'Choose a new output directory: {output}')
    cfg = replace(cfg, validate_geometry=int_config.enable_safeguards)
    first_iteration = 1
    if continuation is not None:
        first_iteration = continuation.next_iteration
        if cfg.max_iterations < first_iteration:
            raise ValueError(f'max_iterations ({cfg.max_iterations}) must reach at least update '
                             f'{first_iteration} to continue {continuation.source}')
        cfg = replace(cfg, seed=continuation.seed)
    initial = load_mesh(input_mesh, validate_geometry=cfg.validate_geometry)
    volume_target = (signed_volume(initial.coordinates, initial.faces) if continuation is None
                     else continuation.volume_target)
    if cfg.validate_geometry:
        validate_domain(initial, initial.coordinates, cfg)
    write_displaced_h5m(initial, None, initial.coordinates, int_config, dry_run=True)
    openmc.reset_auto_ids()
    model = build_model(cfg, initial.path)
    if cfg.validate_geometry:
        validate_model_container(model, cfg)
    tally = objective_tally(model)
    source = model_source_path(cfg)
    library = Path(openmc.lib._dll._name).resolve()
    provenance = dict(method='int_inspired', model_variant=cfg.model_variant,
                      model_sha256=file_hash(source), library_sha256=file_hash(library),
                      model_file=str(source), library=str(library), openmc_version=openmc.__version__,
                      objective_tally=dict(id=tally.id, name=tally.name, scores=list(tally.scores),
                                           estimator=tally.estimator or 'auto',
                                           filters_xml=[etree.tostring(f.to_xml_element(), encoding='unicode')
                                                        for f in tally.filters]),
                      acceptance='completed_INT_update_and_transport; observed change is diagnostic',
                      inner_reference='original run volume target; gradient refreshed each outer update',
                      volume_target=volume_target,
                      visualization='shared surface_mesh plots, report, and vtk_output services')
    if continuation is not None:
        changed = [name for name in ('model', 'library') if provenance[f'{name}_sha256']
                   != continuation.provenance[f'source_{name}_sha256']]
        if changed and not allow_changed_model:
            raise RuntimeError(f"The {' and '.join(changed)} changed since {continuation.source}; "
                               'pass allow_changed_model to continue anyway')
        provenance['continued_from'] = dict(continuation.provenance, changed=changed)
    history = History(output, initial.path, cfg, mode='int_inspired')
    history.data.update(int_inspired=asdict(int_config), int_run=asdict(cfg),
                        settings_kind='int', attempted_iterations=first_iteration - 1, iteration_history=[])
    if continuation is not None:
        history.data['continued_from'] = provenance['continued_from']
    history.save()
    write_report(output / 'provenance.json', provenance)
    write_report(output / 'run_settings.json', {'int_run': asdict(cfg)})
    write_report(output / 'int_inspired_config.json', {'int_inspired': asdict(int_config)})
    (output / 'model_source.py').write_bytes(source.read_bytes())
    (output / 'launcher_source.py').write_bytes(Path(__file__).read_bytes())

    def checked_evaluate(*args, **kwargs):
        for path, key in ((source, 'model_sha256'), (library, 'library_sha256')):
            if file_hash(path) != provenance[key]:
                raise RuntimeError(f'Model/library changed during INT run: {path}')
        with phase(f'Evaluating {Path(args[0]).name} (including visualization)'):
            sample = evaluator(*args, **kwargs)
        for path, key in ((source, 'model_sha256'), (library, 'library_sha256')):
            if file_hash(path) != provenance[key]:
                raise RuntimeError(f'Model/library changed during INT run: {path}')
        return sample

    # Units are explicit in the run metadata; no derivative renormalization.
    metadata = dict(tally_id=tally.id, objective=int_config.objective,
                    coordinate_units='cm', derivative_units=f'{tally.scores[0]} tally units / cm',
                    objective_tally=provenance['objective_tally'])
    current_path = history.initial
    adam_state, step_config = None, int_config
    sampling = AdaptiveSampler(cfg.adaptive_sampling)
    evaluation_cfg = cfg
    decisions = []
    if continuation is not None:
        adam_state = continuation.adam_state if int_config.persist_adam_state else None
        step_config = replace(int_config, learning_rate=continuation.learning_rate)
        sampling = continuation.sampler(cfg.adaptive_sampling)
        # Keep sampling increases only while adaptive sampling stays enabled.
        if cfg.adaptive_sampling.enabled:
            evaluation_cfg = replace(cfg, particles=continuation.particles)
        decisions = [dict(d) for d in continuation.decisions]
    history.data['adaptive_sampling'] = dict(observations=sampling.observations, decisions=decisions)
    status = 'iteration_limit'
    print(f'INT run: {output}\nModel: {cfg.model_variant}; objective: {tally.name}\n'
          f'{cfg.max_iterations} outer updates, {int_config.iterations} Adam steps per update', flush=True)
    if continuation is not None:
        info = continuation.provenance
        print(f"Continuing {continuation.source.name} after update {info['last_accepted_iteration']}: "
              f"Adam state {info['adam_state']} ({info['adam_steps']} steps), learning rate "
              f"{step_config.learning_rate:g}, {evaluation_cfg.particles:,} particles/batch, "
              f"seeds from {cfg.seed}", flush=True)
    if cfg.adaptive_sampling.enabled:
        print(f'[INT sampling] Monitoring from update {cfg.adaptive_sampling.start_iteration}; '
              f'{cfg.adaptive_sampling.growth_factor:g}x increases, '
              f'cap {cfg.adaptive_sampling.max_particles:,} particles/batch; '
              f'{cfg.batches} batches unchanged.', flush=True)
    try:
        current, current_eval = history.evaluate(current_path, evaluation_cfg, checked_evaluate)
        baseline = float(current['tally_mean'])
        if continuation is None:
            sampling.observe(0, current_eval, current)
        else:
            # A fresh sample of the source's last accepted mesh, like a sampling refresh.
            history.data['evaluations'][current_eval]['purpose'] = 'continuation_baseline'
        history.save()
        with phase('Writing baseline animation'):
            history.update_animation()
            render_motion(output)
        for iteration in range(first_iteration, cfg.max_iterations + 1):
            decision = sampling.decide(evaluation_cfg.particles)
            history.data['adaptive_sampling']['decisions'].append(decision)
            if decision['increase']:
                evaluation_cfg = replace(evaluation_cfg, particles=decision['particles_after'])
                print(f"[INT sampling] After update {iteration - 1}: {', '.join(decision['reasons'])}; "
                      f"particles/batch {decision['particles_before']:,} -> "
                      f"{decision['particles_after']:,}. Refreshing current mesh before update {iteration}.",
                      flush=True)
                history.save()
                current, current_eval = history.evaluate(current_path, evaluation_cfg, checked_evaluate)
                history.data['evaluations'][current_eval]['purpose'] = 'adaptive_sampling_refresh'
                decision['resampling_evaluation'] = current_eval
                decision['status'] = 'refreshed'
                history.save()
            mesh = load_mesh(current_path, validate_geometry=cfg.validate_geometry)
            sample_path = output / history.data['evaluations'][current_eval]['directory'] / 'sample.npz'
            bundle = load_derivatives(sample_path, snapshot=mesh, legacy_metadata=metadata)
            name = f'iteration_{iteration:04d}_trial_01'
            record = dict(iteration=iteration, parent_mesh=history.relative(current_path),
                          gradient_evaluation=current_eval, status='optimizing', method='int_inspired',
                          learning_rate=step_config.learning_rate)
            history.data['trials'].append(record)
            history.data['attempted_iterations'] = iteration
            history.save()
            with phase(f'INT update {iteration}/{cfg.max_iterations}: {int_config.iterations} Adam steps'):
                result = optimize_vertices(bundle.vertices, bundle.faces, bundle.gradients, step_config,
                                           volume_target=volume_target, adam_state=adam_state)
            if int_config.persist_adam_state:
                adam_state = result.adam_state
                # Saved before transport; a continuation uses only accepted updates' states.
                history.arrays(f'{name}_adam.npz', **adam_state)
                record['adam_state'] = f'{name}_adam.npz'
            final = result.diagnostics['final']
            print(f"[INT progress] Motion: max {final['max_displacement']:.6g} cm; "
                  f"after subtracting mean translation {final['max_displacement_without_translation']:.6g} cm; "
                  f"volume change {final['relative_volume_error']:+.3%}", flush=True)
            ratio = final['volume_to_transport_gradient_norm_ratio']
            if ratio is not None:
                print(f'[INT progress] Final Cartesian volume/transport gradient norm ratio: {ratio:.3g}',
                      flush=True)
            write_report(output / f'{name}_int.json', dict(status=result.status, history=result.history,
                                                          diagnostics=result.diagnostics, metadata=bundle.metadata))
            record['int_history'] = f'{name}_int.json'
            if result.status == 'rejected_step':
                record.update(status='geometry_rejected', reason=result.history[-1]['reason'])
                status = 'geometry_rejected'
                break
            if np.array_equal(result.vertices, mesh.coordinates):
                record['status'] = 'stationary'
                status = 'stationary'
                break
            candidate_path = output / f'{name}.h5m'
            try:
                if cfg.validate_geometry:
                    validate_domain(mesh, result.vertices, cfg)
                metrics = write_displaced_h5m(mesh, candidate_path, result.vertices, int_config)
            except ValueError as exc:
                record.update(status='geometry_rejected', reason=str(exc))
                status = 'geometry_rejected'
                break
            candidate = load_mesh(candidate_path, validate_geometry=cfg.validate_geometry)
            applied = candidate.coordinates - mesh.coordinates
            predicted = float(np.sum(bundle.gradients * applied))
            record.update(mesh=history.relative(candidate_path), status='evaluating',
                          predicted_change=predicted, metrics=dict(metrics, predicted_change=predicted))
            arrays = dict(coordinates=candidate.coordinates, gradient=bundle.gradients,
                          direction=applied, applied_displacement=applied,
                          faces=mesh.faces, vertex_handles=mesh.vertex_handles)
            history.arrays(f'{name}.npz', **arrays)
            history.trial_vectors(record, current_path, candidate_path, arrays)
            if int_config.remesh_interval and iteration % int_config.remesh_interval == 0:
                remeshed_path = output / f'{name}_remeshed.h5m'
                record['status'] = 'remeshing'
                history.save()
                remesh_config = int_config
                if iteration // int_config.remesh_interval in int_config.remesh_halve_edge_rounds:
                    target = int_config.remesh_target_edge_length
                    if target is None:
                        f = candidate.faces
                        edges = np.concatenate((f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]))
                        x = candidate.coordinates
                        target = float(np.linalg.norm(x[edges[:, 0]] - x[edges[:, 1]], axis=1).mean())
                    remesh_config = replace(int_config, remesh_target_edge_length=.5 * target)
                with phase(f'Remeshing update {iteration} before fresh transport'):
                    remeshing = remesh_h5m(candidate, remeshed_path, remesh_config)
                # INT discards Adam and rescales its step to the new edge length.
                if int_config.remesh_adam_state == 'reset':
                    adam_state = None
                if int_config.remesh_learning_rate_per_edge is not None:
                    step_config = replace(step_config, learning_rate=(
                        int_config.remesh_learning_rate_per_edge * remeshing['target_edge_length']))
                remeshing.update(adam_state=int_config.remesh_adam_state,
                                 next_learning_rate=step_config.learning_rate)
                remeshed = load_mesh(remeshed_path, validate_geometry=cfg.validate_geometry)
                remeshing['relative_volume_error_from_initial'] = (
                    remeshing['volume_after'] - volume_target) / volume_target
                if cfg.validate_geometry:
                    validate_domain(remeshed, remeshed.coordinates, cfg)
                record.update(optimization_mesh=history.relative(candidate_path),
                              mesh=history.relative(remeshed_path), remeshing=remeshing, status='evaluating',
                              prediction_scope='Adam displacement before remeshing')
                candidate_path = remeshed_path
                print(f"[INT progress] Remeshed: {remeshing['vertices_before']} -> "
                      f"{remeshing['vertices_after']} vertices; volume change "
                      f"{remeshing['volume_change']:+.6g} cm^3; refreshing derivatives", flush=True)
            history.save()
            candidate_sample, candidate_eval = history.evaluate(candidate_path, evaluation_cfg, checked_evaluate)
            observed = float(candidate_sample['tally_mean'] - current['tally_mean'])
            sign = 1 if int_config.objective == 'minimize' else -1
            record.update(status='accepted', candidate_evaluation=candidate_eval,
                          observed_tally_change=observed, actual_improvement=-sign * observed,
                          difference_std_dev=float(np.hypot(current['tally_std_dev'],
                                                           candidate_sample['tally_std_dev'])))
            history.data['accepted_sequence'].append(record['mesh'])
            history.data['accepted_mesh'] = record['mesh']
            history.data['iteration_history'].append(dict(iteration=iteration, status='accepted',
                                                          final_evaluation=candidate_eval,
                                                          accepted_mesh=record['mesh']))
            current_path, current, current_eval = candidate_path, candidate_sample, candidate_eval
            sampling.observe(iteration, current_eval, current)
            history.save()
            with phase('Updating accepted-mesh animations'):
                history.update_animation()
                render_motion(output)
            print(f'INT update {iteration}: predicted delta {predicted:.6g}; '
                  f'observed delta {observed:.6g}', flush=True)
        history.finish(status)
        write_report(output / 'completion_summary.json', dict(
            method='int_inspired', status=status, initial_tally_mean=baseline,
            final_tally_mean=float(current['tally_mean']),
            accepted_steps=len(history.data['accepted_sequence']) - 1,
            evaluations=len(history.data['evaluations']), final_mesh=history.data['accepted_mesh'],
            final_particles=evaluation_cfg.particles,
            acceptance=provenance['acceptance'],
            continued_from=history.data.get('continued_from')))
    except Exception as exc:
        history.finish('failed', error=str(exc))
        raise
    return history.data
