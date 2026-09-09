# ML Experiment Tracker - Lightweight Experiment Management
# Author: Maharshi Soni | License: MIT
"""
Pydantic models for experiment tracking entities.
Provides validation and serialization for experiments, runs, metrics,
parameters, artifacts, and tags.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


class RunStatus(str, Enum):
    """Status of an experiment run."""

    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    KILLED = "killed"


class Parameter(BaseModel):
    """A hyperparameter logged for a run."""

    key: str = Field(..., min_length=1, description="Parameter name")
    value: str = Field(..., description="Parameter value (stored as string)")
    run_id: Optional[str] = Field(default=None, description="Associated run ID")

    @field_validator("value", mode="before")
    @classmethod
    def coerce_value_to_str(cls, v: Any) -> str:
        """Coerce parameter values to strings for storage."""
        return str(v)


class Metric(BaseModel):
    """A metric data point logged during a run."""

    key: str = Field(..., min_length=1, description="Metric name")
    value: float = Field(..., description="Metric value")
    step: int = Field(default=0, ge=0, description="Training step or epoch")
    timestamp: Optional[datetime] = Field(
        default=None, description="When the metric was logged"
    )
    run_id: Optional[str] = Field(default=None, description="Associated run ID")


class Artifact(BaseModel):
    """An artifact (file/object) associated with a run."""

    name: str = Field(..., min_length=1, description="Artifact name")
    artifact_type: str = Field(
        default="file", description="Type of artifact (file, model, dataset, plot)"
    )
    path: str = Field(..., description="Storage path for the artifact")
    size_bytes: Optional[int] = Field(
        default=None, ge=0, description="File size in bytes"
    )
    metadata: Dict[str, str] = Field(
        default_factory=dict, description="Additional metadata"
    )
    run_id: Optional[str] = Field(default=None, description="Associated run ID")


class Tag(BaseModel):
    """A tag for organizing experiments and runs."""

    key: str = Field(..., min_length=1, description="Tag key")
    value: str = Field(default="", description="Tag value")


class Run(BaseModel):
    """A single experiment run with parameters, metrics, and artifacts."""

    run_id: str = Field(..., description="Unique run identifier")
    experiment_id: str = Field(..., description="Parent experiment ID")
    run_name: Optional[str] = Field(default=None, description="Human-readable run name")
    status: RunStatus = Field(default=RunStatus.RUNNING, description="Run status")
    start_time: datetime = Field(
        default_factory=datetime.utcnow, description="Run start time"
    )
    end_time: Optional[datetime] = Field(default=None, description="Run end time")
    parameters: List[Parameter] = Field(
        default_factory=list, description="Logged hyperparameters"
    )
    metrics: List[Metric] = Field(
        default_factory=list, description="Logged metrics"
    )
    artifacts: List[Artifact] = Field(
        default_factory=list, description="Logged artifacts"
    )
    tags: List[Tag] = Field(default_factory=list, description="Run tags")
    notes: Optional[str] = Field(default=None, description="Free-form run notes")

    @property
    def duration_seconds(self) -> Optional[float]:
        """Calculate run duration in seconds."""
        if self.end_time and self.start_time:
            return (self.end_time - self.start_time).total_seconds()
        return None


class Experiment(BaseModel):
    """An experiment that groups related runs."""

    experiment_id: str = Field(..., description="Unique experiment identifier")
    name: str = Field(..., min_length=1, description="Experiment name")
    description: Optional[str] = Field(
        default=None, description="Experiment description"
    )
    created_at: datetime = Field(
        default_factory=datetime.utcnow, description="Creation timestamp"
    )
    tags: List[Tag] = Field(default_factory=list, description="Experiment-level tags")
    runs: List[Run] = Field(default_factory=list, description="Associated runs")
