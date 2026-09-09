# ML Experiment Tracker - Lightweight Experiment Management
# Author: Maharshi Soni | License: MIT
"""
High-level API for experiment tracking.
Provides a convenient context-manager interface for logging experiments,
parameters, metrics, and artifacts during ML training runs.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from tracker.models import Artifact, RunStatus, Tag
from tracker.store import ExperimentStore


def _generate_id() -> str:
    """Generate a short unique identifier."""
    return uuid.uuid4().hex[:12]


class ExperimentTracker:
    """High-level interface for tracking ML experiments.

    Usage::

        tracker = ExperimentTracker("experiments.db")
        exp = tracker.create_experiment("mnist-cnn", "CNN experiments on MNIST")

        with tracker.start_run(exp.experiment_id, run_name="lr-0.001") as run_id:
            tracker.log_params(run_id, {"lr": 0.001, "batch_size": 32})
            for epoch in range(10):
                tracker.log_metrics(run_id, {"loss": ..., "acc": ...}, step=epoch)
    """

    def __init__(self, db_path: str = "experiments.db") -> None:
        self.store = ExperimentStore(db_path)
        self._active_run_id: Optional[str] = None

    def close(self) -> None:
        """Close the underlying store."""
        self.store.close()

    # -- Experiment management --

    def create_experiment(
        self,
        name: str,
        description: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
    ) -> str:
        """Create a new experiment and return its ID."""
        experiment_id = _generate_id()
        tag_list = [Tag(key=k, value=v) for k, v in (tags or {}).items()]
        self.store.create_experiment(
            experiment_id=experiment_id,
            name=name,
            description=description,
            tags=tag_list,
        )
        return experiment_id

    def get_or_create_experiment(
        self, name: str, description: Optional[str] = None
    ) -> str:
        """Get an existing experiment by name, or create one."""
        for exp in self.store.list_experiments():
            if exp.name == name:
                return exp.experiment_id
        return self.create_experiment(name, description)

    def list_experiments(self) -> List[Dict[str, Any]]:
        """List all experiments as dictionaries."""
        experiments = self.store.list_experiments()
        return [
            {
                "experiment_id": e.experiment_id,
                "name": e.name,
                "description": e.description,
                "created_at": e.created_at.isoformat(),
                "tags": {t.key: t.value for t in e.tags},
            }
            for e in experiments
        ]

    # -- Run management --

    def start_run(
        self,
        experiment_id: str,
        run_name: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
        notes: Optional[str] = None,
    ) -> _RunContext:
        """Start a new run and return a context manager.

        The run is automatically ended when the context manager exits.
        """
        run_id = _generate_id()
        tag_list = [Tag(key=k, value=v) for k, v in (tags or {}).items()]
        self.store.create_run(
            run_id=run_id,
            experiment_id=experiment_id,
            run_name=run_name,
            tags=tag_list,
            notes=notes,
        )
        self._active_run_id = run_id
        return _RunContext(self, run_id)

    def end_run(
        self, run_id: str, status: RunStatus = RunStatus.COMPLETED
    ) -> None:
        """Manually end a run."""
        self.store.end_run(run_id, status)
        if self._active_run_id == run_id:
            self._active_run_id = None

    def get_run(self, run_id: str) -> Optional[Dict[str, Any]]:
        """Get a run as a dictionary."""
        run = self.store.get_run(run_id)
        if run is None:
            return None
        return {
            "run_id": run.run_id,
            "experiment_id": run.experiment_id,
            "run_name": run.run_name,
            "status": run.status.value,
            "start_time": run.start_time.isoformat(),
            "end_time": run.end_time.isoformat() if run.end_time else None,
            "duration_seconds": run.duration_seconds,
            "parameters": {p.key: p.value for p in run.parameters},
            "metrics": {m.key: m.value for m in run.metrics},
            "artifacts": [
                {"name": a.name, "type": a.artifact_type, "path": a.path}
                for a in run.artifacts
            ],
            "tags": {t.key: t.value for t in run.tags},
            "notes": run.notes,
        }

    def list_runs(
        self,
        experiment_id: Optional[str] = None,
        status: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """List runs with optional filters."""
        run_status = RunStatus(status) if status else None
        runs = self.store.list_runs(experiment_id=experiment_id, status=run_status)
        return [
            {
                "run_id": r.run_id,
                "experiment_id": r.experiment_id,
                "run_name": r.run_name,
                "status": r.status.value,
                "start_time": r.start_time.isoformat(),
                "end_time": r.end_time.isoformat() if r.end_time else None,
                "duration_seconds": r.duration_seconds,
            }
            for r in runs
        ]

    # -- Logging --

    def log_param(self, run_id: str, key: str, value: Any) -> None:
        """Log a single parameter."""
        self.store.log_parameter(run_id, key, value)

    def log_params(self, run_id: str, params: Dict[str, Any]) -> None:
        """Log multiple parameters."""
        self.store.log_parameters(run_id, params)

    def log_metric(
        self, run_id: str, key: str, value: float, step: int = 0
    ) -> None:
        """Log a single metric."""
        self.store.log_metric(run_id, key, value, step)

    def log_metrics(
        self, run_id: str, metrics: Dict[str, float], step: int = 0
    ) -> None:
        """Log multiple metrics at a given step."""
        self.store.log_metrics(run_id, metrics, step)

    def log_artifact(
        self,
        run_id: str,
        name: str,
        path: str,
        artifact_type: str = "file",
        size_bytes: Optional[int] = None,
        metadata: Optional[Dict[str, str]] = None,
    ) -> None:
        """Log an artifact for a run."""
        self.store.log_artifact(
            run_id=run_id,
            name=name,
            path=path,
            artifact_type=artifact_type,
            size_bytes=size_bytes,
            metadata=metadata,
        )

    # -- Comparison & Export --

    def compare_runs(
        self,
        run_ids: List[str],
        metric_keys: Optional[List[str]] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """Compare multiple runs side-by-side."""
        return self.store.compare_runs(run_ids, metric_keys)

    def get_metric_history(
        self, run_id: str, metric_key: str
    ) -> List[Dict[str, Any]]:
        """Get step-by-step metric history."""
        history = self.store.get_metric_history(run_id, metric_key)
        return [{"step": step, "value": value} for step, value in history]

    def export_to_csv(
        self,
        output_path: str,
        experiment_id: Optional[str] = None,
    ) -> str:
        """Export runs to CSV."""
        rows = self.store.export_runs_as_dicts(experiment_id)
        df = pd.DataFrame(rows)
        df.to_csv(output_path, index=False)
        return output_path

    def export_to_json(
        self,
        output_path: str,
        experiment_id: Optional[str] = None,
    ) -> str:
        """Export runs to JSON."""
        rows = self.store.export_runs_as_dicts(experiment_id)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(rows, f, indent=2, default=str)
        return output_path


class _RunContext:
    """Context manager for a training run."""

    def __init__(self, tracker: ExperimentTracker, run_id: str) -> None:
        self.tracker = tracker
        self.run_id = run_id

    def __enter__(self) -> str:
        return self.run_id

    def __exit__(
        self,
        exc_type: Optional[type],
        exc_val: Optional[BaseException],
        exc_tb: Optional[Any],
    ) -> None:
        status = RunStatus.FAILED if exc_type else RunStatus.COMPLETED
        self.tracker.end_run(self.run_id, status)
