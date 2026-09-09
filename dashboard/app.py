# ML Experiment Tracker - Lightweight Experiment Management
# Author: Maharshi Soni | License: MIT
"""
Flask web dashboard for experiment visualization and comparison.
Provides routes for viewing experiments, runs, metrics, and comparison charts.
All charts are rendered as base64-encoded matplotlib images.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from flask import Flask, render_template, request, jsonify, redirect, url_for

from tracker.store import ExperimentStore
from tracker.visualizations import (
    plot_metric_history,
    plot_metric_comparison,
    plot_parameter_vs_metric,
    plot_run_summary_bar,
)

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def create_app(db_path: str = "experiments.db") -> Flask:
    """Create and configure the Flask application."""
    app = Flask(
        __name__,
        template_folder=os.path.join(_BASE_DIR, "templates"),
    )
    app.config["DB_PATH"] = db_path

    def _get_store() -> ExperimentStore:
        return ExperimentStore(app.config["DB_PATH"])

    @app.route("/")
    def index() -> str:
        """Dashboard home: list all experiments."""
        store = _get_store()
        try:
            experiments = store.list_experiments()
            exp_data = []
            for exp in experiments:
                runs = store.list_runs(experiment_id=exp.experiment_id)
                exp_data.append(
                    {
                        "experiment_id": exp.experiment_id,
                        "name": exp.name,
                        "description": exp.description or "",
                        "created_at": exp.created_at.strftime("%Y-%m-%d %H:%M"),
                        "num_runs": len(runs),
                        "tags": {t.key: t.value for t in exp.tags},
                    }
                )
            return render_template("index.html", experiments=exp_data)
        finally:
            store.close()

    @app.route("/experiment/<experiment_id>")
    def experiment_detail(experiment_id: str) -> str:
        """View a single experiment and its runs."""
        store = _get_store()
        try:
            experiment = store.get_experiment(experiment_id)
            if experiment is None:
                return render_template("error.html", message="Experiment not found"), 404
            runs = store.list_runs(experiment_id=experiment_id)
            run_data = []
            for run in runs:
                # Get latest metrics for each run
                latest_metrics: Dict[str, float] = {}
                seen_keys: set[str] = set()
                for m in run.metrics:
                    seen_keys.add(m.key)
                for key in seen_keys:
                    val = store.get_latest_metric(run.run_id, key)
                    if val is not None:
                        latest_metrics[key] = round(val, 6)
                run_data.append(
                    {
                        "run_id": run.run_id,
                        "run_name": run.run_name or run.run_id[:8],
                        "status": run.status.value,
                        "start_time": run.start_time.strftime("%Y-%m-%d %H:%M"),
                        "duration": (
                            f"{run.duration_seconds:.1f}s"
                            if run.duration_seconds
                            else "N/A"
                        ),
                        "parameters": {p.key: p.value for p in run.parameters},
                        "metrics": latest_metrics,
                        "num_artifacts": len(run.artifacts),
                    }
                )
            return render_template(
                "experiment.html",
                experiment={
                    "experiment_id": experiment.experiment_id,
                    "name": experiment.name,
                    "description": experiment.description or "",
                    "created_at": experiment.created_at.strftime("%Y-%m-%d %H:%M"),
                    "tags": {t.key: t.value for t in experiment.tags},
                },
                runs=run_data,
            )
        finally:
            store.close()

    @app.route("/run/<run_id>")
    def run_detail(run_id: str) -> str:
        """View a single run with metric charts."""
        store = _get_store()
        try:
            run = store.get_run(run_id)
            if run is None:
                return render_template("error.html", message="Run not found"), 404

            # Generate metric charts
            metric_keys: set[str] = set()
            for m in run.metrics:
                metric_keys.add(m.key)

            charts = {}
            for key in sorted(metric_keys):
                charts[key] = plot_metric_history(store, run_id, key)

            run_data = {
                "run_id": run.run_id,
                "experiment_id": run.experiment_id,
                "run_name": run.run_name or run.run_id[:8],
                "status": run.status.value,
                "start_time": run.start_time.strftime("%Y-%m-%d %H:%M:%S"),
                "end_time": (
                    run.end_time.strftime("%Y-%m-%d %H:%M:%S") if run.end_time else "N/A"
                ),
                "duration": (
                    f"{run.duration_seconds:.1f}s" if run.duration_seconds else "N/A"
                ),
                "parameters": {p.key: p.value for p in run.parameters},
                "artifacts": [
                    {
                        "name": a.name,
                        "type": a.artifact_type,
                        "path": a.path,
                        "size": (
                            f"{a.size_bytes / 1024:.1f} KB"
                            if a.size_bytes
                            else "N/A"
                        ),
                    }
                    for a in run.artifacts
                ],
                "tags": {t.key: t.value for t in run.tags},
                "notes": run.notes,
            }
            return render_template("run.html", run=run_data, charts=charts)
        finally:
            store.close()

    @app.route("/compare", methods=["GET", "POST"])
    def compare_runs() -> str:
        """Compare multiple runs side-by-side with charts."""
        store = _get_store()
        try:
            if request.method == "POST":
                run_ids = request.form.getlist("run_ids")
            else:
                run_ids_param = request.args.get("run_ids", "")
                run_ids = [r.strip() for r in run_ids_param.split(",") if r.strip()]

            if not run_ids:
                # Show experiment/run selector
                experiments = store.list_experiments()
                all_runs = []
                for exp in experiments:
                    runs = store.list_runs(experiment_id=exp.experiment_id)
                    for run in runs:
                        all_runs.append(
                            {
                                "run_id": run.run_id,
                                "run_name": run.run_name or run.run_id[:8],
                                "experiment_name": exp.name,
                                "status": run.status.value,
                            }
                        )
                return render_template("compare_select.html", runs=all_runs)

            # Build comparison data
            comparison = store.compare_runs(run_ids)
            run_labels = {
                rid: data.get("run_name", rid[:8])
                for rid, data in comparison.items()
            }

            # Collect all metric keys across runs
            all_metric_keys: set[str] = set()
            for data in comparison.values():
                all_metric_keys.update(data.get("metrics", {}).keys())

            # Generate comparison charts
            charts = {}
            for key in sorted(all_metric_keys):
                charts[f"{key}_line"] = plot_metric_comparison(
                    store, run_ids, key, run_labels
                )
                charts[f"{key}_bar"] = plot_run_summary_bar(
                    store, run_ids, key, run_labels
                )

            # Generate param vs metric scatter plots
            all_param_keys: set[str] = set()
            for data in comparison.values():
                all_param_keys.update(data.get("parameters", {}).keys())

            scatter_charts = {}
            for param_key in sorted(all_param_keys):
                for metric_key in sorted(all_metric_keys):
                    chart_key = f"{param_key}_vs_{metric_key}"
                    scatter_charts[chart_key] = plot_parameter_vs_metric(
                        store, run_ids, param_key, metric_key
                    )

            return render_template(
                "compare.html",
                comparison=comparison,
                run_labels=run_labels,
                charts=charts,
                scatter_charts=scatter_charts,
                metric_keys=sorted(all_metric_keys),
                param_keys=sorted(all_param_keys),
            )
        finally:
            store.close()

    @app.route("/api/experiments")
    def api_experiments() -> Any:
        """API endpoint: list experiments."""
        store = _get_store()
        try:
            experiments = store.list_experiments()
            return jsonify(
                [
                    {
                        "experiment_id": e.experiment_id,
                        "name": e.name,
                        "description": e.description,
                        "created_at": e.created_at.isoformat(),
                    }
                    for e in experiments
                ]
            )
        finally:
            store.close()

    @app.route("/api/runs/<experiment_id>")
    def api_runs(experiment_id: str) -> Any:
        """API endpoint: list runs for an experiment."""
        store = _get_store()
        try:
            runs = store.list_runs(experiment_id=experiment_id)
            return jsonify(
                [
                    {
                        "run_id": r.run_id,
                        "run_name": r.run_name,
                        "status": r.status.value,
                        "start_time": r.start_time.isoformat(),
                    }
                    for r in runs
                ]
            )
        finally:
            store.close()

    @app.route("/api/run/<run_id>/metrics")
    def api_run_metrics(run_id: str) -> Any:
        """API endpoint: get metrics for a run."""
        store = _get_store()
        try:
            run = store.get_run(run_id)
            if run is None:
                return jsonify({"error": "Run not found"}), 404
            metric_keys: set[str] = set()
            for m in run.metrics:
                metric_keys.add(m.key)
            result: Dict[str, List[Dict[str, Any]]] = {}
            for key in metric_keys:
                history = store.get_metric_history(run_id, key)
                result[key] = [{"step": s, "value": v} for s, v in history]
            return jsonify(result)
        finally:
            store.close()

    return app
