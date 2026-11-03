from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def tetrahedron():
    x = np.array([[0., 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
    faces = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]])
    return x, faces


@pytest.fixture
def sphere_path():
    return Path(__file__).resolve().parents[1] / "sphere.h5m"
