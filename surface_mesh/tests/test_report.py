"""Check report selection, trial visibility, and composite GIF contents."""

import csv
import json

import numpy as np
from PIL import Image

from ..core.plots import update_accepted_animation
from ..core.report import accepted_evaluations, save_objective_history


def example_history(root):
    entries = []
    for index, (mesh, mean) in enumerate([
            ('initial.h5m', 1.1), ('first.h5m', .9), ('initial.h5m', 1.),
            ('unaccepted.h5m', .7), ('second.h5m', .8)]):
        folder = root / f'evaluation_{index:04d}'
        folder.mkdir()
        for filename, color in [('geometry.png', (240, 30, 30)),
                                ('geometry_solid.png', (30, 220, 30))]:
            with Image.new('RGB', (120, 100), color) as image:
                image.save(folder / filename)
        entries.append(dict(index=index, mesh=mesh, tally_mean=mean, tally_std_dev=.01,
                            status='complete', image=f'{folder.name}/geometry.png',
                            solid_image=f'{folder.name}/geometry_solid.png', seed=index, budget=1))
    entries.append(dict(index=5, mesh='failed.h5m', status='failed'))
    return dict(status='sampling_limited', settings={'animation_duration_ms': 100},
                accepted_sequence=['initial.h5m', 'first.h5m', 'second.h5m'],
                evaluations=entries, trials=[dict(mesh='unaccepted.h5m', status='inconclusive')])


def test_history_keeps_all_evaluations_and_refines_accepted_estimates(tmp_path):
    history = example_history(tmp_path)
    assert [(step, entry['index']) for step, entry in accepted_evaluations(history)] == [(0, 2), (1, 1), (2, 4)]
    save_objective_history(tmp_path, history)
    with (tmp_path / 'objective_history.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 5
    assert rows[3]['accepted_geometry'] == 'False'
    assert rows[3]['trial_status'] == 'inconclusive'
    assert rows[0]['tally_mean'] == '1.1' and rows[2]['tally_mean'] == '1.0'
    with Image.open(tmp_path / 'objective_history.png') as image:
        image.verify()
    assert '<svg' in (tmp_path / 'objective_history.svg').read_text(encoding='utf-8')


def test_composite_contains_slices_render_and_only_accepted_frames(tmp_path, monkeypatch):
    history = example_history(tmp_path)
    # Simulate a resampled record without images: reuse this mesh's earlier images.
    history['evaluations'][2].pop('image')
    history['evaluations'][2].pop('solid_image')
    monkeypatch.setattr('surface_mesh.core.plots.ANIMATION_WIDTH', 1200)
    update_accepted_animation(tmp_path, history)
    with Image.open(tmp_path / 'accepted_geometry.gif') as image:
        assert image.n_frames == 3
        pixels = np.asarray(image.convert('RGB')).astype(int)
        assert np.any(np.max(abs(pixels - (240, 30, 30)), axis=2) < 10)
        assert np.any(np.max(abs(pixels - (30, 220, 30)), axis=2) < 10)
    records = json.loads((tmp_path / 'animation_frames.json').read_text())
    assert [record['mesh'] for record in records] == history['accepted_sequence']
    assert records[0]['evaluation'] == 2
    assert records[0]['solid_image'] == 'evaluation_0000/geometry_solid.png'
