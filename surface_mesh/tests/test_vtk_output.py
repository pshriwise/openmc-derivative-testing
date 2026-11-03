"""Read exported diagnostics with VTK itself and compare against saved meshes."""

import json
from pathlib import Path

import numpy as np
import pytest
from surface_mesh.mesh import load_mesh

vtk = pytest.importorskip('vtk')
from vtk.util.numpy_support import vtk_to_numpy

from ..core.vtk_output import export_trial

INPUT_MESH = Path(__file__).resolve().parents[1].joinpath('sphere_coarse.h5m')


def read_vtk(path):
    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(str(path))
    reader.ReadAllVectorsOn()
    reader.ReadAllScalarsOn()
    reader.Update()
    mesh = reader.GetOutput()
    data = mesh.GetPointData()
    arrays = {data.GetArrayName(i): vtk_to_numpy(data.GetArray(i)).copy()
              for i in range(data.GetNumberOfArrays())}
    return mesh, arrays


def test_wrong_displacement_is_rejected(tmp_path):
    mesh = load_mesh(INPUT_MESH)
    arrays = dict(gradient=np.ones_like(mesh.coordinates),
                  direction=np.zeros_like(mesh.coordinates),
                  applied_displacement=np.ones_like(mesh.coordinates))
    with pytest.raises(ValueError, match='does not reproduce'):
        export_trial(INPUT_MESH, INPUT_MESH, arrays, tmp_path / 'bad.vtk')
    assert not (tmp_path / 'bad.vtk').exists()


def test_zero_vectors_have_explicitly_undefined_alignment(tmp_path):
    mesh = load_mesh(INPUT_MESH)
    zeros = np.zeros_like(mesh.coordinates)
    path = tmp_path / 'zero.vtk'
    export_trial(INPUT_MESH, INPUT_MESH,
                 dict(gradient=zeros, direction=zeros, applied_displacement=zeros), path)
    _, arrays = read_vtk(path)
    assert not arrays['descent_alignment_defined'].any()
    assert np.isfinite(arrays['descent_alignment']).all()


def test_series_updates_incrementally_with_animation(tmp_path):
    from ..core.history import History

    history = History.__new__(History)
    history.root = tmp_path
    history.data = dict(settings={'save_vtk': True}, accepted_sequence=['initial.h5m'],
                        evaluations=[dict(mesh='initial.h5m', status='complete', vtk='initial.vtk')])
    (tmp_path / 'initial.vtk').touch()
    # Isolate the shared animation hook from objective plotting.
    from unittest.mock import patch
    with patch('surface_mesh.core.report.save_objective_history'):
        history.update_animation()
        series_path = tmp_path / 'mesh_evolution.vtk.series'
        assert json.loads(series_path.read_text())['files'] == [
            dict(name='initial.vtk', time=0)]
        (tmp_path / 'step1.vtk').touch()
        history.data['accepted_sequence'].append('step1.h5m')
        history.data['evaluations'].append(dict(mesh='step1.h5m', status='complete', vtk='step1.vtk'))
        history.update_animation()
        assert json.loads(series_path.read_text())['files'] == [
            dict(name='initial.vtk', time=0), dict(name='step1.vtk', time=1)]
