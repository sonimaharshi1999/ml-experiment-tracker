# ML Experiment Tracker - Lightweight Experiment Management
# Author: Maharshi Soni | License: MIT
"""Tests for visualization utilities."""

from __future__ import annotations

import base64
import os

import pytest

from tracker.store import ExperimentStore
from tracker.visualizations import (
    plot_metric_comparison,
    plot_metric_history,
    plot_parameter_vs_metric,
    plot_run_summary_bar,
)


@pytest.fixture
def store_with_runs(tmp_path: object) -> ExperimentStore:
    """Create a store with sample runs for visualization tests."""
    db_path = os.path.join(str(tmp_path), "test_viz.db")
    store = ExperimentStore(db_path)
    store.create_experiment("exp-viz", "Viz Test")

    for i, lr in enumerate([0.001, 0.01, 0.1]):
        run_id = f"viz-run-{i}"
        store.create_run(run_id, "exp-viz", run_name=f"lr-{lr}")
        store.log_parameter(run_id, "learning_rate", str(lr))
        for step in range(10):
            store.log_metric(run_id, "loss", 1.0 / (step + 1 + i), step=step)
            store.log_metric(run_id, "accuracy", min(0.5 + step * 0.05 + i * 0.1, 1.0), step=step)
        store.end_run(run_id)

    return store


def _is_valid_base64_png(data: str) -> bool:
    """Check if a string is valid base64 PNG data."""
    try:
        decoded = base64.b64decode(data)
        return decoded[:8] == b"\x89PNG\r\n\x1a\n"
    except Exception:
        return False


class TestVisualizations:
    """Tests for chart generation."""

    def test_plot_metric_history(self, store_with_runs: ExperimentStore) -> None:
        """Test single-run metric history chart."""
        result = plot_metric_history(store_with_runs, "viz-run-0", "loss")
        assert _is_valid_base64_png(result)

    def test_plot_metric_comparison(self, store_with_runs: ExperimentStore) -> None:
        """Test multi-run metric comparison chart."""
        result = plot_metric_comparison(
            store_with_runs,
            ["viz-run-0", "viz-run-1", "viz-run-2"],
            "loss",
        )
        assert _is_valid_base64_png(result)

    def test_plot_parameter_vs_metric(self, store_with_runs: ExperimentStore) -> None:
        """Test parameter vs metric scatter plot."""
        result = plot_parameter_vs_metric(
            store_with_runs,
            ["viz-run-0", "viz-run-1", "viz-run-2"],
            "learning_rate",
            "accuracy",
        )
        assert _is_valid_base64_png(result)

    def test_plot_run_summary_bar(self, store_with_runs: ExperimentStore) -> None:
        """Test bar chart of final metric values."""
        result = plot_run_summary_bar(
            store_with_runs,
            ["viz-run-0", "viz-run-1", "viz-run-2"],
            "accuracy",
        )
        assert _is_valid_base64_png(result)

    def test_empty_metric_returns_placeholder(self, store_with_runs: ExperimentStore) -> None:
        """Test that missing metric data returns a valid placeholder chart."""
        result = plot_metric_history(store_with_runs, "viz-run-0", "nonexistent")
        assert _is_valid_base64_png(result)
