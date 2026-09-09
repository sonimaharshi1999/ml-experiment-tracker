# ML Experiment Tracker - Lightweight Experiment Management
# Author: Maharshi Soni | License: MIT
"""Tests for Pydantic models."""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from tracker.models import (
    Artifact,
    Experiment,
    Metric,
    Parameter,
    Run,
    RunStatus,
    Tag,
)


class TestParameter:
    """Tests for the Parameter model."""

    def test_create_parameter(self) -> None:
        """Test creating a parameter with string value."""
        param = Parameter(key="learning_rate", value="0.001")
        assert param.key == "learning_rate"
        assert param.value == "0.001"

    def test_parameter_coerces_numeric_to_string(self) -> None:
        """Test that numeric values are coerced to strings."""
        param = Parameter(key="batch_size", value=32)  # type: ignore[arg-type]
        assert param.value == "32"
        assert isinstance(param.value, str)

    def test_parameter_coerces_bool_to_string(self) -> None:
        """Test that boolean values are coerced to strings."""
        param = Parameter(key="use_augmentation", value=True)  # type: ignore[arg-type]
        assert param.value == "True"


class TestMetric:
    """Tests for the Metric model."""

    def test_create_metric(self) -> None:
        """Test creating a metric."""
        metric = Metric(key="loss", value=0.5, step=10)
        assert metric.key == "loss"
        assert metric.value == 0.5
        assert metric.step == 10

    def test_metric_default_step(self) -> None:
        """Test that step defaults to 0."""
        metric = Metric(key="accuracy", value=0.95)
        assert metric.step == 0


class TestRun:
    """Tests for the Run model."""

    def test_run_duration(self) -> None:
        """Test run duration calculation."""
        start = datetime(2024, 1, 1, 0, 0, 0)
        end = start + timedelta(hours=1, minutes=30)
        run = Run(
            run_id="abc123",
            experiment_id="exp1",
            start_time=start,
            end_time=end,
            status=RunStatus.COMPLETED,
        )
        assert run.duration_seconds == 5400.0

    def test_run_duration_none_when_running(self) -> None:
        """Test that duration is None for running experiments."""
        run = Run(
            run_id="abc123",
            experiment_id="exp1",
            status=RunStatus.RUNNING,
        )
        assert run.duration_seconds is None


class TestArtifact:
    """Tests for the Artifact model."""

    def test_create_artifact(self) -> None:
        """Test creating an artifact with metadata."""
        artifact = Artifact(
            name="model.pt",
            artifact_type="model",
            path="/models/model.pt",
            size_bytes=1024,
            metadata={"format": "pytorch"},
        )
        assert artifact.name == "model.pt"
        assert artifact.artifact_type == "model"
        assert artifact.metadata["format"] == "pytorch"
