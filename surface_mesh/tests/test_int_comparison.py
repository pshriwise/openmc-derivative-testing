"""Compare reproducible INT variants with a Cartesian reference."""


import numpy as np
import pytest

pytest.importorskip('mitsuba')
pytest.importorskip('cholespy')

from surface_mesh.int_optimizer import IntConfig
from surface_mesh.int_optimizer.comparison import compare_methods
from surface_mesh.int_optimizer.fixtures import sphere, gradient_field


@pytest.mark.parametrize('case', ['localized', 'smooth', 'radial', 'zero'])
def test_comparison_reproducible(tmp_path, case):
    x, f = sphere(1)
    g = gradient_field(x, case)
    cfg = IntConfig(iterations=2)
    reports = [compare_methods(x, f, g, cfg, tmp_path / str(i)) for i in range(2)]
    assert len(reports[0]['methods']) == 3
    for name in reports[0]['methods']:
        first, second = (r['methods'][name]['metrics'] for r in reports)
        assert first['predicted_tally_change'] == pytest.approx(second['predicted_tally_change'], abs=1e-6)
    if case == 'zero':
        assert all(m['metrics']['max_displacement'] == 0 for m in reports[0]['methods'].values())
    np.testing.assert_array_equal(x, sphere(1)[0])
