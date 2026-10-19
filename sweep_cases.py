"""Validate named sweep configurations before starting transport."""

from collections.abc import Mapping
from numbers import Integral, Real
import re

import numpy as np


def validate_sweep_options(parameter, bounds, num_values, iron=None, tally=None):
    """Return normalized sweep options without modifying the supplied mappings."""
    if parameter not in ('r', 'x0', 'y0', 'z0'):
        raise ValueError('Sweep parameter must be r, x0, y0, or z0.')
    if isinstance(num_values, bool) or not isinstance(num_values, Integral) or num_values < 2:
        raise ValueError('num_values must be an integer of at least two.')
    if bounds is not None:
        try:
            bounds = np.asarray(bounds, dtype=float)
        except (TypeError, ValueError) as error:
            raise ValueError('bounds must contain two finite, increasing values.') from error
        if bounds.shape != (2,) or not np.all(np.isfinite(bounds)) or bounds[0] >= bounds[1]:
            raise ValueError('bounds must contain two finite, increasing values.')
        if parameter == 'r' and bounds[0] <= 0:
            raise ValueError('Radius sweep bounds must be positive.')
        bounds = tuple(float(value) for value in bounds)
    iron = {} if iron is None else iron
    tally = {} if tally is None else tally
    if not isinstance(iron, Mapping) or set(iron) - {'r', 'x0', 'y0', 'z0'}:
        raise ValueError('iron must map r, x0, y0, or z0 to initial values.')
    for name, value in iron.items():
        if (isinstance(value, bool) or not isinstance(value, Real)
                or not np.isfinite(value) or (name == 'r' and value <= 0)):
            raise ValueError(f'Invalid iron coefficient {name!r}: {value!r}.')
    if not isinstance(tally, Mapping) or set(tally) - {'score'}:
        raise ValueError('tally supports only the score option.')
    if 'score' in tally and (not isinstance(tally['score'], str) or not tally['score'].strip()):
        raise ValueError('tally.score must be a nonempty OpenMC score name.')
    return {
        'parameter': parameter,
        'bounds': bounds,
        'num_values': int(num_values),
        'iron': {name: float(value) for name, value in iron.items()},
        'tally': {'score': tally['score'].strip()} if 'score' in tally else {},
    }


def validate_cases(cases, case_names=None):
    """Validate every selected case and preserve selection order."""
    if not isinstance(cases, Mapping):
        raise ValueError('SWEEP_CASES must be a dictionary of named cases.')
    names = list(cases) if case_names is None else list(case_names)
    if not names or len(set(names)) != len(names):
        raise ValueError('Select at least one case, with no duplicate names.')
    normalized = {}
    for name in names:
        if not isinstance(name, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]*', name):
            raise ValueError('Case names must start with a letter or digit and contain only letters, digits, _ or -.')
        if name not in cases:
            raise ValueError(f'Unknown sweep case {name!r}. Available: {", ".join(cases)}')
        case = cases[name]
        if not isinstance(case, Mapping) or set(case) - {'parameter', 'bounds', 'num_values', 'iron', 'tally'}:
            raise ValueError(f'Case {name!r} contains unsupported sweep options.')
        if 'parameter' not in case:
            raise ValueError(f'Case {name!r} requires parameter.')
        try:
            normalized[name] = validate_sweep_options(
                case['parameter'], case.get('bounds'), case.get('num_values', 100),
                case.get('iron'), case.get('tally'),
            )
        except ValueError as error:
            raise ValueError(f'Case {name!r}: {error}') from error
    return normalized
