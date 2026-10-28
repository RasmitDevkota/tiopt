import sys
from pathlib import Path

import pytest

REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY / "python"))

import tiopt  # noqa: E402


@pytest.fixture(scope="session")
def ion():
    return tiopt.Ion("40Ca+")


@pytest.fixture(scope="session")
def solved_trap():
    """The linear surface trap preset (with a grating coupler) with its electrodynamics solved analytically."""
    trap = tiopt.Trap.from_preset(7, 50.0, gratings=True)
    trap.solve("analytical")
    return trap


@pytest.fixture
def repository():
    return REPOSITORY
