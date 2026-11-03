"""Incremental manifests, immutable trial meshes, and numerical histories."""

from dataclasses import asdict
import json
from pathlib import Path
import shutil

import numpy as np
from surface_mesh.mesh.reporting import file_hash

from .runtime import atomic_write


class History:
    def __init__(self, output, input_mesh, settings, mode='int_inspired'):
        self.root = Path(output).resolve()
        self.root.mkdir(parents=True, exist_ok=False)
        self.initial = self.root / 'initial.h5m'
        shutil.copyfile(input_mesh, self.initial)
        self.data = {'schema_version': 1, 'mode': mode, 'settings': json.loads(json.dumps(asdict(settings))),
                     'initial_mesh': 'initial.h5m', 'input_sha256': file_hash(self.initial),
                     'accepted_mesh': 'initial.h5m', 'accepted_sequence': ['initial.h5m'],
                     'status': 'running', 'evaluations': [], 'trials': []}
        self.save()

    def relative(self, path):
        return str(Path(path).resolve().relative_to(self.root))

    def save(self):
        if self.data['settings'].get('save_vtk', True):
            from .vtk_output import export_series
            export_series(self.root, self.data)
        encoded = (json.dumps(self.data, indent=2, allow_nan=False) + '\n').encode()
        atomic_write(self.root / 'history.json', lambda stream: stream.write(encoded))

    def evaluate(self, mesh, settings, evaluator, *, budget=1, seed=None):
        index = len(self.data['evaluations'])
        directory = self.root / f'evaluation_{index:04d}'
        record = {'index': index, 'mesh': self.relative(mesh),
                  'directory': directory.name, 'seed': settings.seed + 104729 * index if seed is None else seed,
                  'budget': budget, 'status': 'running'}
        record.update(particles=settings.particles * budget, requested_batches=settings.batches)
        self.data['evaluations'].append(record)
        self.save()
        try:
            sample = evaluator(mesh, directory, settings, record['seed'], budget)
            record.update(status='complete', tally_mean=float(sample['tally_mean']),
                          tally_std_dev=float(sample['tally_std_dev']))
            record['num_realizations'] = int(sample.get('num_realizations', settings.batches))
            if 'geometry_image' in sample:
                record['image'] = self.relative(directory / str(sample['geometry_image']))
            if 'geometry_solid_image' in sample:
                record['solid_image'] = self.relative(directory / str(sample['geometry_solid_image']))
            if settings.save_vtk:
                from .vtk_output import export_evaluation
                output = directory / 'derivatives.vtk'
                export_evaluation(mesh, sample, output,
                                  validate_geometry=getattr(settings, 'validate_geometry', True))
                record['vtk'] = self.relative(output)
            self.save()
            from .report import save_objective_history
            save_objective_history(self.root, self.data)
            return sample, index
        except Exception as exc:
            record.update(status='failed', error=str(exc))
            self.save()
            raise

    def arrays(self, name, **arrays):
        atomic_write(self.root / name,
                     lambda stream: np.savez_compressed(stream, **arrays))

    def finish(self, status, **values):
        self.data.update(status=status, **values)
        self.save()
        self.update_animation()

    def trial_vectors(self, record, parent, candidate, arrays):
        if self.data['settings'].get('save_vtk', True):
            from .vtk_output import export_trial
            output = Path(candidate).with_name(Path(candidate).stem + '_vectors.vtk')
            export_trial(parent, candidate, arrays, output,
                         validate_geometry=self.data['settings'].get('validate_geometry', True))
            record['vtk'] = self.relative(output)

    def update_animation(self):
        if self.data['settings'].get('save_vtk', True):
            from .vtk_output import export_series
            export_series(self.root, self.data)
        from .report import save_objective_history
        save_objective_history(self.root, self.data)
        if (self.data['settings'].get('save_geometry_images')
                or self.data['settings'].get('save_solid_geometry_images')):
            from .plots import update_accepted_animation
            update_accepted_animation(self.root, self.data)
