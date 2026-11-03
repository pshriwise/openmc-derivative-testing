"""Source exclusions shared by the sphere and box deformation workflows."""

from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest

from ..core.settings import IntRunSettings
from ..core.models import build_model, source_position_in_mesh


@pytest.mark.parametrize('variant', ['sphere', 'box'])
def test_source_position_matches_model(variant):
    import openmc
    openmc.reset_auto_ids()
    cfg = IntRunSettings(model_variant=variant)
    model = build_model(cfg, Path(__file__).resolve().parents[1].joinpath('sphere_coarse.h5m'))
    cell = model.geometry.root_universe.cells[20]
    np.testing.assert_array_equal(source_position_in_mesh(cfg),
                                  np.asarray(model.settings.source[0].space.xyz) - cell.translation)
    assert cfg.source_clearance == 0.5
    assert IntRunSettings(**asdict(cfg)).source_clearance == 0.5


@pytest.mark.parametrize('radius', [-0.1, np.nan, np.inf])
def test_invalid_source_clearance(radius):
    with pytest.raises(ValueError, match='source_clearance'):
        IntRunSettings(source_clearance=radius)
