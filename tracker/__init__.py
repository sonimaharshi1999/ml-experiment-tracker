# ML Experiment Tracker - Lightweight Experiment Management
# Author: Maharshi Soni | License: MIT
"""
ML Experiment Tracker: A lightweight, zero-config experiment tracking system
for machine learning workflows. Log hyperparameters, metrics, artifacts, and
model metadata across training runs.
"""

from tracker.models import Experiment, Run, Metric, Parameter, Artifact, Tag
from tracker.store import ExperimentStore
from tracker.tracker import ExperimentTracker

__version__ = "1.0.0"
__author__ = "Maharshi Soni"
__all__ = [
    "Experiment",
    "Run",
    "Metric",
    "Parameter",
    "Artifact",
    "Tag",
    "ExperimentStore",
    "ExperimentTracker",
]
