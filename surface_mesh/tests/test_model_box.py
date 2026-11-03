"""Box/source partitioning and matching deformation-bound checks."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import openmc
import pytest

from ..core.settings import IntRunSettings
from ..core.evaluate import validate_model_container, validate_surface_derivative
from ..core.model_box import build_model, INPUT_MESH
from ..core.models import model_source_path, objective_tally


@pytest.fixture
def model():
    openmc.reset_auto_ids()
    return build_model()


def test_boxes_and_real_air_source_partition(model):
    cells = model.geometry.root_universe.cells
    np.testing.assert_allclose(cells[20].region.bounding_box, [[-1, -13, -13], [20, 13, 13]])
    np.testing.assert_allclose(cells[21].region.bounding_box, [[20, -5, -5], [30, 5, 5]])
    np.testing.assert_allclose(cells[23].region.bounding_box, [[-.1]*3, [.1]*3])
    assert cells[23].fill.name == 'air' and cells[23].fill.density > 0
    assert sum(isinstance(s, openmc.Sphere) for s in model.geometry.get_all_surfaces().values()) == 1
    for point, expected in [((0, 0, 0), 23), ((.11, 0, 0), 20), ((7, 0, 0), 20),
                            ((25, 0, 0), 21), ((-34, 0, 0), 22)]:
        assert [cell.id for cell in cells.values() if point in cell.region] == [expected]
    for point in np.random.default_rng(42).uniform(-34.9, 34.9, (1000, 3)):
        assert sum(point in cell.region for cell in cells.values()) == 1
    tally = objective_tally(model)
    assert tally.scores == ['current']
    assert isinstance(tally.filters[0], openmc.SurfaceFilter)
    assert isinstance(tally.filters[1], openmc.CellFromFilter)
    assert tally.filters[1].bins.tolist() == [22]
    surface_id, = tally.filters[0].bins
    assert model.geometry.get_all_surfaces()[surface_id].x0 == 20
    validate_model_container(model, IntRunSettings(model_variant='box'))


def test_objective_validation_rejects_multiple_bins_and_wrong_derivative(model):
    tally = objective_tally(model)
    tally.scores = ['current', 'flux']
    with pytest.raises(ValueError, match='one objective bin'):
        objective_tally(model)
    tally.scores = ['current']
    model.geometry_derivatives[0].tally = openmc.Tally()
    with pytest.raises(ValueError, match='reference objective tally'):
        objective_tally(model)


def test_original_flux_objective_supported():
    from ..core.model import build_model as build_sphere
    openmc.reset_auto_ids()
    model = build_sphere(INPUT_MESH)
    tally = objective_tally(model)
    assert tally.scores == ['flux']
    assert tally.estimator == 'collision'


def test_surface_objective_requires_populated_derivative(model):
    sample = {key: np.zeros((3, 3)) for key in ('geometry_derivative', 'dJ', 'datt')}
    with pytest.raises(ValueError, match='not evidence of optimization convergence'):
        validate_surface_derivative(objective_tally(model), sample)
    sample['geometry_derivative'][0, 0] = 1e-12
    validate_surface_derivative(objective_tally(model), sample)


def test_bounds_validation_checks_actual_model(model):
    surfaces = model.geometry.root_universe.cells[20].region.get_surfaces()
    plane = next(s for s in surfaces.values() if isinstance(s, openmc.XPlane) and s.x0 == 20)
    plane.x0 = 19.5
    with pytest.raises(ValueError, match='box bounds differ'):
        validate_model_container(model, IntRunSettings(model_variant='box'))


@pytest.mark.parametrize('kwargs,match', [
    ({'container_upper': (14, 13, 13)}, 'water box')])
def test_overlapping_csg_regions_rejected(kwargs, match):
    openmc.reset_auto_ids()
    with pytest.raises(ValueError, match=match):
        build_model(**kwargs)


def test_expanded_container_preserves_source_partition(model):
    from ..core.settings import load_run_config

    cells = model.geometry.root_universe.cells
    np.testing.assert_allclose(cells[20].region.bounding_box,
                               [[-1, -13, -13], [20, 13, 13]])
    for point, expected in [((-1.01, 0, 0), 22), ((-.99, 0, 0), 20),
                            ((-.11, 0, 0), 20), ((0, 0, 0), 23),
                            ((.11, 0, 0), 20)]:
        assert [c.id for c in cells.values() if point in c.region] == [expected]
    for cfg in (IntRunSettings(model_variant='box'),
                load_run_config(Path(__file__).resolve().parents[1].joinpath('int_run_adaptive.json'))):
        validate_model_container(model, cfg)


def test_variant_defaults_and_serialization(model, tmp_path):
    assert model_source_path(IntRunSettings()).name == 'model.py'
    assert model_source_path(IntRunSettings(model_variant='box')).name == 'model_box.py'
    model.export_to_model_xml(tmp_path / 'model_box.xml')
    xml = (tmp_path / 'model_box.xml').read_text()
    assert str(INPUT_MESH.resolve()) in xml
    assert 'source_air' in xml and 'water_box' in xml and 'dagmc_box' in xml
    with pytest.raises(ValueError, match='Box has insufficient room'):
        replace(IntRunSettings(), box_container_upper=(-12, 13, 13))
