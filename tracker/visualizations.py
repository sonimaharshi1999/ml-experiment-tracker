# ML Experiment Tracker - Lightweight Experiment Management
# Author: Maharshi Soni | License: MIT
"""
Visualization utilities for experiment metrics.
Generates matplotlib charts encoded as base64 strings for embedding
in the Flask web dashboard.
"""

from __future__ import annotations

import base64
import io
from typing import Any, Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from tracker.store import ExperimentStore

# Color palette for multi-run plots
COLORS = [
    "#2196F3",
    "#F44336",
    "#4CAF50",
    "#FF9800",
    "#9C27B0",
    "#00BCD4",
    "#795548",
    "#607D8B",
    "#E91E63",
    "#3F51B5",
]


def _fig_to_base64(fig: plt.Figure) -> str:
    """Convert a matplotlib figure to a base64-encoded PNG string."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight", facecolor="white")
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return encoded


def plot_metric_history(
    store: ExperimentStore,
    run_id: str,
    metric_key: str,
    title: Optional[str] = None,
) -> str:
    """Plot a single metric's history for one run.

    Returns a base64-encoded PNG image.
    """
    history = store.get_metric_history(run_id, metric_key)
    if not history:
        return _empty_chart(f"No data for '{metric_key}'")

    steps = [h[0] for h in history]
    values = [h[1] for h in history]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(steps, values, color=COLORS[0], linewidth=2, marker="o", markersize=4)
    ax.set_xlabel("Step / Epoch", fontsize=11)
    ax.set_ylabel(metric_key, fontsize=11)
    ax.set_title(title or f"{metric_key} over training", fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    fig.tight_layout()
    return _fig_to_base64(fig)


def plot_metric_comparison(
    store: ExperimentStore,
    run_ids: List[str],
    metric_key: str,
    run_labels: Optional[Dict[str, str]] = None,
    title: Optional[str] = None,
) -> str:
    """Plot a metric across multiple runs for comparison.

    Returns a base64-encoded PNG image.
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    has_data = False

    for i, run_id in enumerate(run_ids):
        history = store.get_metric_history(run_id, metric_key)
        if not history:
            continue
        has_data = True
        steps = [h[0] for h in history]
        values = [h[1] for h in history]
        label = (run_labels or {}).get(run_id, run_id[:8])
        color = COLORS[i % len(COLORS)]
        ax.plot(steps, values, color=color, linewidth=2, marker="o", markersize=3, label=label)

    if not has_data:
        plt.close(fig)
        return _empty_chart(f"No data for '{metric_key}'")

    ax.set_xlabel("Step / Epoch", fontsize=11)
    ax.set_ylabel(metric_key, fontsize=11)
    ax.set_title(
        title or f"{metric_key} comparison across runs",
        fontsize=13,
        fontweight="bold",
    )
    ax.legend(fontsize=9, loc="best")
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    fig.tight_layout()
    return _fig_to_base64(fig)


def plot_parameter_vs_metric(
    store: ExperimentStore,
    run_ids: List[str],
    param_key: str,
    metric_key: str,
    title: Optional[str] = None,
) -> str:
    """Scatter plot of a parameter value vs final metric across runs.

    Returns a base64-encoded PNG image.
    """
    param_values: List[float] = []
    metric_values: List[float] = []
    labels: List[str] = []

    for run_id in run_ids:
        run = store.get_run(run_id)
        if run is None:
            continue
        param_val = None
        for p in run.parameters:
            if p.key == param_key:
                try:
                    param_val = float(p.value)
                except ValueError:
                    break
                break
        if param_val is None:
            continue
        metric_val = store.get_latest_metric(run_id, metric_key)
        if metric_val is None:
            continue
        param_values.append(param_val)
        metric_values.append(metric_val)
        labels.append(run.run_name or run_id[:8])

    if not param_values:
        return _empty_chart(f"No data for {param_key} vs {metric_key}")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(param_values, metric_values, c=COLORS[0], s=80, edgecolors="white", linewidth=1.5)
    for i, label in enumerate(labels):
        ax.annotate(
            label,
            (param_values[i], metric_values[i]),
            textcoords="offset points",
            xytext=(5, 5),
            fontsize=8,
        )
    ax.set_xlabel(param_key, fontsize=11)
    ax.set_ylabel(f"Final {metric_key}", fontsize=11)
    ax.set_title(
        title or f"{param_key} vs {metric_key}",
        fontsize=13,
        fontweight="bold",
    )
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return _fig_to_base64(fig)


def plot_run_summary_bar(
    store: ExperimentStore,
    run_ids: List[str],
    metric_key: str,
    run_labels: Optional[Dict[str, str]] = None,
    title: Optional[str] = None,
) -> str:
    """Bar chart comparing final metric values across runs.

    Returns a base64-encoded PNG image.
    """
    names: List[str] = []
    values: List[float] = []

    for run_id in run_ids:
        run = store.get_run(run_id)
        if run is None:
            continue
        val = store.get_latest_metric(run_id, metric_key)
        if val is None:
            continue
        label = (run_labels or {}).get(run_id, run.run_name or run_id[:8])
        names.append(label)
        values.append(val)

    if not names:
        return _empty_chart(f"No data for '{metric_key}'")

    fig, ax = plt.subplots(figsize=(max(6, len(names) * 1.2), 5))
    bars = ax.bar(names, values, color=COLORS[: len(names)], edgecolor="white", linewidth=0.8)
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{val:.4f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax.set_ylabel(metric_key, fontsize=11)
    ax.set_title(
        title or f"Final {metric_key} by run",
        fontsize=13,
        fontweight="bold",
    )
    ax.grid(True, axis="y", alpha=0.3)
    plt.xticks(rotation=30, ha="right")
    fig.tight_layout()
    return _fig_to_base64(fig)


def _empty_chart(message: str) -> str:
    """Generate a placeholder chart with a message."""
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.text(
        0.5,
        0.5,
        message,
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=14,
        color="#999999",
    )
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout()
    return _fig_to_base64(fig)
