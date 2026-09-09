# ML Experiment Tracker - Lightweight Experiment Management
# Author: Maharshi Soni | License: MIT
"""Tests for the high-level ExperimentTracker API."""

from __future__ import annotations

import json
import os
from typing import Any, Dict

import pytest

from tracker.models import RunStatus
from tracker.tracker import ExperimentTracker


class TestExperimentTracker:
    """Tests for ExperimentTracker high-level API."""

    def test_create_experiment(self, tracker: ExperimentTracker) -> None:
        """Test creating an experiment returns a valid ID."""
        exp_id = tracker.create_experiment(
            name="test-exp", description="Test experiment"
        )
        assert isinstance(exp_id, str)
        assert len(exp_id) == 12

    def test_get_or_create_experiment(self, tracker: ExperimentTracker) -> None:
        """Test get_or_create returns existing experiment."""
        exp_id1 = tracker.create_experiment("my-exp")
        exp_id2 = tracker.get_or_create_experiment("my-exp")
        assert exp_id1 == exp_id2

    def test_get_or_create_experiment_new(self, tracker: ExperimentTracker) -> None:
        """Test get_or_create creates new experiment when not found."""
        exp_id = tracker.get_or_create_experiment("brand-new")
        assert isinstance(exp_id, str)
        experiments = tracker.list_experiments()
        assert any(e["name"] == "brand-new" for e in experiments)

    def test_run_context_manager_completed(self, tracker: ExperimentTracker) -> None:
        """Test that context manager sets status to COMPLETED on success."""
        exp_id = tracker.create_experiment("ctx-test")
        with tracker.start_run(exp_id, run_name="ctx-run") as run_id:
            tracker.log_param(run_id, "lr", 0.01)
            tracker.log_metric(run_id, "loss", 0.5, step=0)

        run = tracker.get_run(run_id)
        assert run is not None
        assert run["status"] == "completed"

    def test_run_context_manager_failed_on_exception(
        self, tracker: ExperimentTracker
    ) -> None:
        """Test that context manager sets status to FAILED on exception."""
        exp_id = tracker.create_experiment("fail-test")
        run_id = None
        with pytest.raises(ValueError):
            with tracker.start_run(exp_id, run_name="fail-run") as run_id:
                tracker.log_param(run_id, "lr", 0.01)
                raise ValueError("Training failed!")

        assert run_id is not None
        run = tracker.get_run(run_id)
        assert run is not None
        assert run["status"] == "failed"

    def test_log_and_retrieve_metrics(self, tracker: ExperimentTracker) -> None:
        """Test logging metrics and retrieving history."""
        exp_id = tracker.create_experiment("metric-test")
        with tracker.start_run(exp_id) as run_id:
            for step in range(10):
                tracker.log_metrics(
                    run_id, {"loss": 1.0 - step * 0.1, "acc": step * 0.1}, step=step
                )

        history = tracker.get_metric_history(run_id, "loss")
        assert len(history) == 10
        assert history[0]["step"] == 0
        assert abs(history[0]["value"] - 1.0) < 1e-6
        assert history[9]["step"] == 9
        assert abs(history[9]["value"] - 0.1) < 1e-6

    def test_export_to_csv(self, tracker: ExperimentTracker, tmp_path: Any) -> None:
        """Test exporting runs to CSV."""
        exp_id = tracker.create_experiment("csv-test")
        with tracker.start_run(exp_id, run_name="csv-run") as run_id:
            tracker.log_params(run_id, {"lr": 0.01, "batch": 32})
            tracker.log_metric(run_id, "loss", 0.5, step=0)

        csv_path = os.path.join(str(tmp_path), "export.csv")
        result = tracker.export_to_csv(csv_path, exp_id)
        assert os.path.exists(result)
        with open(result, "r") as f:
            content = f.read()
        assert "run_id" in content
        assert "param.lr" in content

    def test_export_to_json(self, tracker: ExperimentTracker, tmp_path: Any) -> None:
        """Test exporting runs to JSON."""
        exp_id = tracker.create_experiment("json-test")
        with tracker.start_run(exp_id, run_name="json-run") as run_id:
            tracker.log_params(run_id, {"lr": 0.01})
            tracker.log_metric(run_id, "loss", 0.3, step=0)

        json_path = os.path.join(str(tmp_path), "export.json")
        result = tracker.export_to_json(json_path, exp_id)
        assert os.path.exists(result)
        with open(result, "r") as f:
            data = json.load(f)
        assert isinstance(data, list)
        assert len(data) == 1
        assert data[0]["run_name"] == "json-run"

    def test_compare_runs(self, tracker: ExperimentTracker) -> None:
        """Test comparing multiple runs."""
        exp_id = tracker.create_experiment("compare-test")
        run_ids = []
        for i in range(3):
            with tracker.start_run(exp_id, run_name=f"run-{i}") as run_id:
                tracker.log_params(run_id, {"lr": 0.001 * (i + 1)})
                tracker.log_metric(run_id, "accuracy", 0.8 + i * 0.05, step=0)
                run_ids.append(run_id)

        comparison = tracker.compare_runs(run_ids)
        assert len(comparison) == 3
        for rid in run_ids:
            assert rid in comparison
            assert "parameters" in comparison[rid]
            assert "metrics" in comparison[rid]

    def test_log_artifact(self, tracker: ExperimentTracker) -> None:
        """Test logging an artifact."""
        exp_id = tracker.create_experiment("artifact-test")
        with tracker.start_run(exp_id, run_name="artifact-run") as run_id:
            tracker.log_artifact(
                run_id,
                name="model.pt",
                path="/models/model.pt",
                artifact_type="model",
                size_bytes=4096,
                metadata={"framework": "pytorch"},
            )

        run = tracker.get_run(run_id)
        assert run is not None
        assert len(run["artifacts"]) == 1
        assert run["artifacts"][0]["name"] == "model.pt"
