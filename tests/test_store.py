# ML Experiment Tracker - Lightweight Experiment Management
# Author: Maharshi Soni | License: MIT
"""Tests for the SQLite experiment store."""

from __future__ import annotations

from typing import List

import pytest

from tracker.models import RunStatus, Tag
from tracker.store import ExperimentStore


class TestExperimentStore:
    """Tests for ExperimentStore CRUD operations."""

    def test_create_and_get_experiment(self, store: ExperimentStore) -> None:
        """Test creating and retrieving an experiment."""
        exp = store.create_experiment(
            experiment_id="exp-001",
            name="test-experiment",
            description="A test experiment",
            tags=[Tag(key="team", value="ml")],
        )
        assert exp.experiment_id == "exp-001"
        assert exp.name == "test-experiment"

        fetched = store.get_experiment("exp-001")
        assert fetched is not None
        assert fetched.name == "test-experiment"
        assert fetched.description == "A test experiment"
        assert len(fetched.tags) == 1
        assert fetched.tags[0].key == "team"

    def test_list_experiments(self, store: ExperimentStore) -> None:
        """Test listing all experiments."""
        store.create_experiment("exp-a", "Experiment A")
        store.create_experiment("exp-b", "Experiment B")
        store.create_experiment("exp-c", "Experiment C")
        experiments = store.list_experiments()
        assert len(experiments) == 3
        names = {e.name for e in experiments}
        assert names == {"Experiment A", "Experiment B", "Experiment C"}

    def test_delete_experiment(self, store: ExperimentStore) -> None:
        """Test deleting an experiment removes it and its runs."""
        store.create_experiment("exp-del", "To Delete")
        store.create_run("run-del", "exp-del")
        store.log_parameter("run-del", "lr", "0.01")
        store.log_metric("run-del", "loss", 0.5, step=0)

        result = store.delete_experiment("exp-del")
        assert result is True
        assert store.get_experiment("exp-del") is None
        assert store.get_run("run-del") is None

    def test_create_and_get_run(self, store: ExperimentStore) -> None:
        """Test creating and retrieving a run."""
        store.create_experiment("exp-1", "Test Exp")
        run = store.create_run(
            run_id="run-001",
            experiment_id="exp-1",
            run_name="first-run",
            tags=[Tag(key="trial", value="1")],
        )
        assert run.run_id == "run-001"
        assert run.status == RunStatus.RUNNING

        fetched = store.get_run("run-001")
        assert fetched is not None
        assert fetched.run_name == "first-run"
        assert len(fetched.tags) == 1

    def test_end_run(self, store: ExperimentStore) -> None:
        """Test ending a run updates its status."""
        store.create_experiment("exp-1", "Test Exp")
        store.create_run("run-end", "exp-1")
        store.end_run("run-end", RunStatus.COMPLETED)

        run = store.get_run("run-end")
        assert run is not None
        assert run.status == RunStatus.COMPLETED
        assert run.end_time is not None

    def test_log_and_retrieve_parameters(self, store: ExperimentStore) -> None:
        """Test logging and retrieving parameters."""
        store.create_experiment("exp-1", "Test Exp")
        store.create_run("run-p", "exp-1")
        store.log_parameters(
            "run-p", {"lr": 0.001, "batch_size": 32, "optimizer": "adam"}
        )

        run = store.get_run("run-p")
        assert run is not None
        params = {p.key: p.value for p in run.parameters}
        assert params["lr"] == "0.001"
        assert params["batch_size"] == "32"
        assert params["optimizer"] == "adam"

    def test_log_and_retrieve_metrics(self, store: ExperimentStore) -> None:
        """Test logging and retrieving metrics with step history."""
        store.create_experiment("exp-1", "Test Exp")
        store.create_run("run-m", "exp-1")

        for step in range(5):
            store.log_metrics(
                "run-m",
                {"loss": 1.0 / (step + 1), "accuracy": (step + 1) * 0.2},
                step=step,
            )

        history = store.get_metric_history("run-m", "loss")
        assert len(history) == 5
        assert history[0] == (0, 1.0)
        assert history[4][0] == 4
        assert abs(history[4][1] - 0.2) < 1e-6

        latest = store.get_latest_metric("run-m", "accuracy")
        assert latest is not None
        assert abs(latest - 1.0) < 1e-6

    def test_log_and_retrieve_artifacts(self, store: ExperimentStore) -> None:
        """Test logging and retrieving artifacts."""
        store.create_experiment("exp-1", "Test Exp")
        store.create_run("run-a", "exp-1")
        store.log_artifact(
            run_id="run-a",
            name="model.pt",
            path="/artifacts/model.pt",
            artifact_type="model",
            size_bytes=2048,
            metadata={"format": "pytorch"},
        )

        run = store.get_run("run-a")
        assert run is not None
        assert len(run.artifacts) == 1
        assert run.artifacts[0].name == "model.pt"
        assert run.artifacts[0].size_bytes == 2048
        assert run.artifacts[0].metadata["format"] == "pytorch"

    def test_compare_runs(self, store: ExperimentStore) -> None:
        """Test comparing multiple runs."""
        store.create_experiment("exp-1", "Test Exp")

        for i, lr in enumerate(["0.001", "0.01", "0.1"]):
            run_id = f"run-cmp-{i}"
            store.create_run(run_id, "exp-1", run_name=f"lr-{lr}")
            store.log_parameter(run_id, "lr", lr)
            store.log_metric(run_id, "accuracy", 0.9 - i * 0.1, step=0)
            store.end_run(run_id, RunStatus.COMPLETED)

        comparison = store.compare_runs(
            ["run-cmp-0", "run-cmp-1", "run-cmp-2"]
        )
        assert len(comparison) == 3
        assert comparison["run-cmp-0"]["parameters"]["lr"] == "0.001"
        assert abs(comparison["run-cmp-0"]["metrics"]["accuracy"] - 0.9) < 1e-6

    def test_list_runs_with_status_filter(self, store: ExperimentStore) -> None:
        """Test filtering runs by status."""
        store.create_experiment("exp-1", "Test Exp")
        store.create_run("run-ok", "exp-1")
        store.end_run("run-ok", RunStatus.COMPLETED)
        store.create_run("run-fail", "exp-1")
        store.end_run("run-fail", RunStatus.FAILED)
        store.create_run("run-active", "exp-1")

        completed = store.list_runs(
            experiment_id="exp-1", status=RunStatus.COMPLETED
        )
        assert len(completed) == 1
        assert completed[0].run_id == "run-ok"

        running = store.list_runs(
            experiment_id="exp-1", status=RunStatus.RUNNING
        )
        assert len(running) == 1

    def test_export_runs_as_dicts(self, store: ExperimentStore) -> None:
        """Test exporting runs as flat dictionaries."""
        store.create_experiment("exp-1", "Test Exp")
        store.create_run("run-exp", "exp-1", run_name="export-test")
        store.log_parameter("run-exp", "lr", "0.01")
        store.log_metric("run-exp", "loss", 0.5, step=0)
        store.end_run("run-exp", RunStatus.COMPLETED)

        rows = store.export_runs_as_dicts("exp-1")
        assert len(rows) == 1
        assert rows[0]["run_id"] == "run-exp"
        assert rows[0]["param.lr"] == "0.01"
        assert abs(rows[0]["metric.loss"] - 0.5) < 1e-6
