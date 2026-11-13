import os

# Headless plotting for CI and servers.
os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np  # noqa: E402
import pytest  # noqa: E402


@pytest.fixture
def toy_matrix():
    """Small symmetric contact matrix with a strong block in the middle."""
    rng = np.random.default_rng(0)
    n = 40
    m = rng.poisson(3, size=(n, n)).astype(np.int32)
    m = np.triu(m) + np.triu(m, 1).T
    m[15:20, 15:20] += 60
    return m
