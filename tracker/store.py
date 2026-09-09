# ML Experiment Tracker - Lightweight Experiment Management
# Author: Maharshi Soni | License: MIT
"""
SQLite-backed storage engine for experiment tracking.
Handles all database operations: creating experiments, logging runs,
storing metrics/parameters/artifacts, and querying data.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from tracker.models import (
    Artifact,
    Experiment,
    Metric,
    Parameter,
    Run,
    RunStatus,
    Tag,
)

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS experiments (
    experiment_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS experiment_tags (
    experiment_id TEXT NOT NULL,
    key TEXT NOT NULL,
    value TEXT NOT NULL DEFAULT '',
    PRIMARY KEY (experiment_id, key),
    FOREIGN KEY (experiment_id) REFERENCES experiments(experiment_id)
);

CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    experiment_id TEXT NOT NULL,
    run_name TEXT,
    status TEXT NOT NULL DEFAULT 'running',
    start_time TEXT NOT NULL,
    end_time TEXT,
    notes TEXT,
    FOREIGN KEY (experiment_id) REFERENCES experiments(experiment_id)
);

CREATE TABLE IF NOT EXISTS run_tags (
    run_id TEXT NOT NULL,
    key TEXT NOT NULL,
    value TEXT NOT NULL DEFAULT '',
    PRIMARY KEY (run_id, key),
    FOREIGN KEY (run_id) REFERENCES runs(run_id)
);

CREATE TABLE IF NOT EXISTS parameters (
    run_id TEXT NOT NULL,
    key TEXT NOT NULL,
    value TEXT NOT NULL,
    PRIMARY KEY (run_id, key),
    FOREIGN KEY (run_id) REFERENCES runs(run_id)
);

CREATE TABLE IF NOT EXISTS metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    key TEXT NOT NULL,
    value REAL NOT NULL,
    step INTEGER NOT NULL DEFAULT 0,
    timestamp TEXT,
    FOREIGN KEY (run_id) REFERENCES runs(run_id)
);

CREATE TABLE IF NOT EXISTS artifacts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    name TEXT NOT NULL,
    artifact_type TEXT NOT NULL DEFAULT 'file',
    path TEXT NOT NULL,
    size_bytes INTEGER,
    metadata TEXT NOT NULL DEFAULT '{}',
    FOREIGN KEY (run_id) REFERENCES runs(run_id)
);

CREATE INDEX IF NOT EXISTS idx_metrics_run_key ON metrics(run_id, key);
CREATE INDEX IF NOT EXISTS idx_runs_experiment ON runs(experiment_id);
CREATE INDEX IF NOT EXISTS idx_runs_status ON runs(status);
"""


class ExperimentStore:
    """SQLite-backed storage for experiment data."""

    def __init__(self, db_path: str = "experiments.db") -> None:
        self.db_path = db_path
        self._conn: Optional[sqlite3.Connection] = None
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        """Get or create a database connection."""
        if self._conn is None:
            self._conn = sqlite3.connect(self.db_path)
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA foreign_keys=ON")
        return self._conn

    def _init_db(self) -> None:
        """Initialize the database schema."""
        conn = self._get_connection()
        conn.executescript(_SCHEMA_SQL)
        conn.commit()

    def close(self) -> None:
        """Close the database connection."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    # -- Experiment CRUD --

    def create_experiment(
        self,
        experiment_id: str,
        name: str,
        description: Optional[str] = None,
        tags: Optional[List[Tag]] = None,
    ) -> Experiment:
        """Create a new experiment."""
        conn = self._get_connection()
        now = datetime.utcnow().isoformat()
        conn.execute(
            "INSERT INTO experiments (experiment_id, name, description, created_at) "
            "VALUES (?, ?, ?, ?)",
            (experiment_id, name, description, now),
        )
        if tags:
            for tag in tags:
                conn.execute(
                    "INSERT OR REPLACE INTO experiment_tags (experiment_id, key, value) "
                    "VALUES (?, ?, ?)",
                    (experiment_id, tag.key, tag.value),
                )
        conn.commit()
        return Experiment(
            experiment_id=experiment_id,
            name=name,
            description=description,
            created_at=datetime.fromisoformat(now),
            tags=tags or [],
        )

    def get_experiment(self, experiment_id: str) -> Optional[Experiment]:
        """Retrieve an experiment by ID."""
        conn = self._get_connection()
        row = conn.execute(
            "SELECT * FROM experiments WHERE experiment_id = ?", (experiment_id,)
        ).fetchone()
        if row is None:
            return None
        tags = self._get_experiment_tags(experiment_id)
        return Experiment(
            experiment_id=row["experiment_id"],
            name=row["name"],
            description=row["description"],
            created_at=datetime.fromisoformat(row["created_at"]),
            tags=tags,
        )

    def list_experiments(self) -> List[Experiment]:
        """List all experiments."""
        conn = self._get_connection()
        rows = conn.execute(
            "SELECT * FROM experiments ORDER BY created_at DESC"
        ).fetchall()
        experiments = []
        for row in rows:
            tags = self._get_experiment_tags(row["experiment_id"])
            experiments.append(
                Experiment(
                    experiment_id=row["experiment_id"],
                    name=row["name"],
                    description=row["description"],
                    created_at=datetime.fromisoformat(row["created_at"]),
                    tags=tags,
                )
            )
        return experiments

    def delete_experiment(self, experiment_id: str) -> bool:
        """Delete an experiment and all its runs."""
        conn = self._get_connection()
        # Get all runs for this experiment
        run_ids = [
            r["run_id"]
            for r in conn.execute(
                "SELECT run_id FROM runs WHERE experiment_id = ?", (experiment_id,)
            ).fetchall()
        ]
        for run_id in run_ids:
            self._delete_run_data(run_id)
        conn.execute(
            "DELETE FROM experiment_tags WHERE experiment_id = ?", (experiment_id,)
        )
        cursor = conn.execute(
            "DELETE FROM experiments WHERE experiment_id = ?", (experiment_id,)
        )
        conn.commit()
        return cursor.rowcount > 0

    def _get_experiment_tags(self, experiment_id: str) -> List[Tag]:
        """Get tags for an experiment."""
        conn = self._get_connection()
        rows = conn.execute(
            "SELECT key, value FROM experiment_tags WHERE experiment_id = ?",
            (experiment_id,),
        ).fetchall()
        return [Tag(key=r["key"], value=r["value"]) for r in rows]

    # -- Run CRUD --

    def create_run(
        self,
        run_id: str,
        experiment_id: str,
        run_name: Optional[str] = None,
        tags: Optional[List[Tag]] = None,
        notes: Optional[str] = None,
    ) -> Run:
        """Create a new run within an experiment."""
        conn = self._get_connection()
        now = datetime.utcnow().isoformat()
        conn.execute(
            "INSERT INTO runs (run_id, experiment_id, run_name, status, start_time, notes) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (run_id, experiment_id, run_name, RunStatus.RUNNING.value, now, notes),
        )
        if tags:
            for tag in tags:
                conn.execute(
                    "INSERT OR REPLACE INTO run_tags (run_id, key, value) "
                    "VALUES (?, ?, ?)",
                    (run_id, tag.key, tag.value),
                )
        conn.commit()
        return Run(
            run_id=run_id,
            experiment_id=experiment_id,
            run_name=run_name,
            status=RunStatus.RUNNING,
            start_time=datetime.fromisoformat(now),
            tags=tags or [],
            notes=notes,
        )

    def end_run(self, run_id: str, status: RunStatus = RunStatus.COMPLETED) -> None:
        """Mark a run as finished."""
        conn = self._get_connection()
        now = datetime.utcnow().isoformat()
        conn.execute(
            "UPDATE runs SET status = ?, end_time = ? WHERE run_id = ?",
            (status.value, now, run_id),
        )
        conn.commit()

    def get_run(self, run_id: str) -> Optional[Run]:
        """Retrieve a run by ID with all its data."""
        conn = self._get_connection()
        row = conn.execute(
            "SELECT * FROM runs WHERE run_id = ?", (run_id,)
        ).fetchone()
        if row is None:
            return None
        return self._build_run(row)

    def list_runs(
        self,
        experiment_id: Optional[str] = None,
        status: Optional[RunStatus] = None,
        tag_key: Optional[str] = None,
        tag_value: Optional[str] = None,
    ) -> List[Run]:
        """List runs with optional filters."""
        conn = self._get_connection()
        query = "SELECT * FROM runs WHERE 1=1"
        params: List[Any] = []
        if experiment_id:
            query += " AND experiment_id = ?"
            params.append(experiment_id)
        if status:
            query += " AND status = ?"
            params.append(status.value)
        if tag_key:
            query += (
                " AND run_id IN (SELECT run_id FROM run_tags WHERE key = ?"
            )
            params.append(tag_key)
            if tag_value:
                query += " AND value = ?"
                params.append(tag_value)
            query += ")"
        query += " ORDER BY start_time DESC"
        rows = conn.execute(query, params).fetchall()
        return [self._build_run(row) for row in rows]

    def _build_run(self, row: sqlite3.Row) -> Run:
        """Build a Run model from a database row."""
        run_id = row["run_id"]
        return Run(
            run_id=run_id,
            experiment_id=row["experiment_id"],
            run_name=row["run_name"],
            status=RunStatus(row["status"]),
            start_time=datetime.fromisoformat(row["start_time"]),
            end_time=(
                datetime.fromisoformat(row["end_time"]) if row["end_time"] else None
            ),
            notes=row["notes"],
            parameters=self._get_parameters(run_id),
            metrics=self._get_metrics(run_id),
            artifacts=self._get_artifacts(run_id),
            tags=self._get_run_tags(run_id),
        )

    def _delete_run_data(self, run_id: str) -> None:
        """Delete all data associated with a run."""
        conn = self._get_connection()
        conn.execute("DELETE FROM metrics WHERE run_id = ?", (run_id,))
        conn.execute("DELETE FROM parameters WHERE run_id = ?", (run_id,))
        conn.execute("DELETE FROM artifacts WHERE run_id = ?", (run_id,))
        conn.execute("DELETE FROM run_tags WHERE run_id = ?", (run_id,))
        conn.execute("DELETE FROM runs WHERE run_id = ?", (run_id,))

    def _get_run_tags(self, run_id: str) -> List[Tag]:
        """Get tags for a run."""
        conn = self._get_connection()
        rows = conn.execute(
            "SELECT key, value FROM run_tags WHERE run_id = ?", (run_id,)
        ).fetchall()
        return [Tag(key=r["key"], value=r["value"]) for r in rows]

    # -- Parameters --

    def log_parameter(self, run_id: str, key: str, value: Any) -> None:
        """Log a parameter for a run."""
        conn = self._get_connection()
        conn.execute(
            "INSERT OR REPLACE INTO parameters (run_id, key, value) VALUES (?, ?, ?)",
            (run_id, key, str(value)),
        )
        conn.commit()

    def log_parameters(self, run_id: str, params: Dict[str, Any]) -> None:
        """Log multiple parameters for a run."""
        conn = self._get_connection()
        for key, value in params.items():
            conn.execute(
                "INSERT OR REPLACE INTO parameters (run_id, key, value) VALUES (?, ?, ?)",
                (run_id, key, str(value)),
            )
        conn.commit()

    def _get_parameters(self, run_id: str) -> List[Parameter]:
        """Get all parameters for a run."""
        conn = self._get_connection()
        rows = conn.execute(
            "SELECT key, value FROM parameters WHERE run_id = ?", (run_id,)
        ).fetchall()
        return [Parameter(key=r["key"], value=r["value"], run_id=run_id) for r in rows]

    # -- Metrics --

    def log_metric(
        self, run_id: str, key: str, value: float, step: int = 0
    ) -> None:
        """Log a metric data point for a run."""
        conn = self._get_connection()
        now = datetime.utcnow().isoformat()
        conn.execute(
            "INSERT INTO metrics (run_id, key, value, step, timestamp) "
            "VALUES (?, ?, ?, ?, ?)",
            (run_id, key, value, step, now),
        )
        conn.commit()

    def log_metrics(
        self, run_id: str, metrics: Dict[str, float], step: int = 0
    ) -> None:
        """Log multiple metrics for a run at a given step."""
        conn = self._get_connection()
        now = datetime.utcnow().isoformat()
        for key, value in metrics.items():
            conn.execute(
                "INSERT INTO metrics (run_id, key, value, step, timestamp) "
                "VALUES (?, ?, ?, ?, ?)",
                (run_id, key, value, step, now),
            )
        conn.commit()

    def _get_metrics(self, run_id: str) -> List[Metric]:
        """Get all metrics for a run."""
        conn = self._get_connection()
        rows = conn.execute(
            "SELECT key, value, step, timestamp FROM metrics "
            "WHERE run_id = ? ORDER BY step",
            (run_id,),
        ).fetchall()
        return [
            Metric(
                key=r["key"],
                value=r["value"],
                step=r["step"],
                timestamp=(
                    datetime.fromisoformat(r["timestamp"]) if r["timestamp"] else None
                ),
                run_id=run_id,
            )
            for r in rows
        ]

    def get_metric_history(
        self, run_id: str, metric_key: str
    ) -> List[Tuple[int, float]]:
        """Get the history of a metric (step, value) for a run."""
        conn = self._get_connection()
        rows = conn.execute(
            "SELECT step, value FROM metrics WHERE run_id = ? AND key = ? ORDER BY step",
            (run_id, metric_key),
        ).fetchall()
        return [(r["step"], r["value"]) for r in rows]

    def get_latest_metric(
        self, run_id: str, metric_key: str
    ) -> Optional[float]:
        """Get the most recent value of a metric for a run."""
        conn = self._get_connection()
        row = conn.execute(
            "SELECT value FROM metrics WHERE run_id = ? AND key = ? "
            "ORDER BY step DESC LIMIT 1",
            (run_id, metric_key),
        ).fetchone()
        return row["value"] if row else None

    # -- Artifacts --

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
        conn = self._get_connection()
        conn.execute(
            "INSERT INTO artifacts (run_id, name, artifact_type, path, size_bytes, metadata) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                run_id,
                name,
                artifact_type,
                path,
                size_bytes,
                json.dumps(metadata or {}),
            ),
        )
        conn.commit()

    def _get_artifacts(self, run_id: str) -> List[Artifact]:
        """Get all artifacts for a run."""
        conn = self._get_connection()
        rows = conn.execute(
            "SELECT name, artifact_type, path, size_bytes, metadata "
            "FROM artifacts WHERE run_id = ?",
            (run_id,),
        ).fetchall()
        return [
            Artifact(
                name=r["name"],
                artifact_type=r["artifact_type"],
                path=r["path"],
                size_bytes=r["size_bytes"],
                metadata=json.loads(r["metadata"]),
                run_id=run_id,
            )
            for r in rows
        ]

    # -- Comparison & Export --

    def compare_runs(
        self, run_ids: List[str], metric_keys: Optional[List[str]] = None
    ) -> Dict[str, Dict[str, Any]]:
        """Compare multiple runs side-by-side.

        Returns a dict keyed by run_id with parameters and latest metrics.
        """
        result: Dict[str, Dict[str, Any]] = {}
        for run_id in run_ids:
            run = self.get_run(run_id)
            if run is None:
                continue
            params = {p.key: p.value for p in run.parameters}
            metrics: Dict[str, float] = {}
            seen_keys: set[str] = set()
            for m in run.metrics:
                seen_keys.add(m.key)
            for key in seen_keys:
                if metric_keys and key not in metric_keys:
                    continue
                latest = self.get_latest_metric(run_id, key)
                if latest is not None:
                    metrics[key] = latest
            result[run_id] = {
                "run_name": run.run_name,
                "status": run.status.value,
                "parameters": params,
                "metrics": metrics,
                "duration_seconds": run.duration_seconds,
            }
        return result

    def export_runs_as_dicts(
        self, experiment_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Export runs as a list of flat dictionaries for CSV/JSON export."""
        runs = self.list_runs(experiment_id=experiment_id)
        rows: List[Dict[str, Any]] = []
        for run in runs:
            row: Dict[str, Any] = {
                "run_id": run.run_id,
                "experiment_id": run.experiment_id,
                "run_name": run.run_name or "",
                "status": run.status.value,
                "start_time": run.start_time.isoformat(),
                "end_time": run.end_time.isoformat() if run.end_time else "",
                "duration_seconds": run.duration_seconds,
            }
            for p in run.parameters:
                row[f"param.{p.key}"] = p.value
            # Use latest metric value for each key
            seen_metric_keys: set[str] = set()
            for m in run.metrics:
                seen_metric_keys.add(m.key)
            for key in sorted(seen_metric_keys):
                latest = self.get_latest_metric(run.run_id, key)
                if latest is not None:
                    row[f"metric.{key}"] = latest
            for tag in run.tags:
                row[f"tag.{tag.key}"] = tag.value
            rows.append(row)
        return rows
