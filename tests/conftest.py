# ML Experiment Tracker - Lightweight Experiment Management
# Author: Maharshi Soni | License: MIT
"""Shared test fixtures."""

from __future__ import annotations

import os
import tempfile
from typing import Generator

import pytest

from tracker.store import ExperimentStore
from tracker.tracker import ExperimentTracker


@pytest.fixture
def tmp_db(tmp_path: object) -> str:
    """Create a temporary database path."""
    return os.path.join(str(tmp_path), "test_experiments.db")


@pytest.fixture
def store(tmp_db: str) -> Generator[ExperimentStore, None, None]:
    """Create a fresh ExperimentStore backed by a temp database."""
    s = ExperimentStore(tmp_db)
    yield s
    s.close()


@pytest.fixture
def tracker(tmp_db: str) -> Generator[ExperimentTracker, None, None]:
    """Create a fresh ExperimentTracker backed by a temp database."""
    t = ExperimentTracker(tmp_db)
    yield t
    t.close()
