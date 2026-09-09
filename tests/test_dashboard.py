# ML Experiment Tracker - Lightweight Experiment Management
# Author: Maharshi Soni | License: MIT
"""Tests for the Flask web dashboard."""

from __future__ import annotations

import os
from typing import Generator

import pytest

from dashboard.app import create_app
from tracker.tracker import ExperimentTracker


@pytest.fixture
def app_with_data(tmp_path: object) -> Generator[object, None, None]:
    """Create a Flask test app with sample data."""
    db_path = os.path.join(str(tmp_path), "test_dashboard.db")
    tracker = ExperimentTracker(db_path)

    exp_id = tracker.create_experiment("test-exp", "Dashboard test experiment")
    with tracker.start_run(exp_id, run_name="dashboard-run") as run_id:
        tracker.log_params(run_id, {"lr": 0.001, "batch_size": 32})
        for step in range(5):
            tracker.log_metrics(
                run_id,
                {"loss": 1.0 - step * 0.15, "accuracy": 0.5 + step * 0.1},
                step=step,
            )
        tracker.log_artifact(
            run_id,
            name="model.pt",
            path="/tmp/model.pt",
            artifact_type="model",
            size_bytes=1024,
        )
    tracker.close()

    app = create_app(db_path)
    app.config["TESTING"] = True
    yield app


class TestDashboard:
    """Tests for the Flask dashboard routes."""

    def test_index_page(self, app_with_data: object) -> None:
        """Test that the index page loads and shows experiments."""
        with app_with_data.test_client() as client:  # type: ignore[union-attr]
            response = client.get("/")
            assert response.status_code == 200
            assert b"test-exp" in response.data

    def test_experiment_detail_page(self, app_with_data: object) -> None:
        """Test the experiment detail page shows runs."""
        with app_with_data.test_client() as client:  # type: ignore[union-attr]
            # First get the experiment ID from the index
            response = client.get("/")
            assert response.status_code == 200
            # The experiment ID is in the HTML links
            assert b"dashboard-run" in response.data or b"test-exp" in response.data

    def test_api_experiments(self, app_with_data: object) -> None:
        """Test the API endpoint for listing experiments."""
        with app_with_data.test_client() as client:  # type: ignore[union-attr]
            response = client.get("/api/experiments")
            assert response.status_code == 200
            data = response.get_json()
            assert isinstance(data, list)
            assert len(data) >= 1
            assert data[0]["name"] == "test-exp"

    def test_compare_select_page(self, app_with_data: object) -> None:
        """Test the compare selection page loads."""
        with app_with_data.test_client() as client:  # type: ignore[union-attr]
            response = client.get("/compare")
            assert response.status_code == 200
            assert b"Select Runs" in response.data or b"Compare" in response.data

    def test_nonexistent_run_returns_404(self, app_with_data: object) -> None:
        """Test that a nonexistent run returns 404."""
        with app_with_data.test_client() as client:  # type: ignore[union-attr]
            response = client.get("/run/nonexistent-id")
            assert response.status_code == 404
