# ML Experiment Tracker - Lightweight Experiment Management
# Author: Maharshi Soni | License: MIT
"""
Command-line interface for the experiment tracker.
Provides commands for listing experiments, viewing runs, comparing runs,
and exporting data to CSV/JSON.
"""

from __future__ import annotations

import json
import sys
from typing import List, Optional

import click

from tracker.tracker import ExperimentTracker


@click.group()
@click.option(
    "--db",
    default="experiments.db",
    envvar="TRACKER_DB",
    help="Path to the SQLite database file.",
)
@click.pass_context
def cli(ctx: click.Context, db: str) -> None:
    """ML Experiment Tracker - manage and analyze experiment runs."""
    ctx.ensure_object(dict)
    ctx.obj["tracker"] = ExperimentTracker(db)


@cli.command("experiments")
@click.pass_context
def list_experiments(ctx: click.Context) -> None:
    """List all experiments."""
    tracker: ExperimentTracker = ctx.obj["tracker"]
    experiments = tracker.list_experiments()
    if not experiments:
        click.echo("No experiments found.")
        return
    click.echo(f"{'ID':<14} {'Name':<30} {'Runs':>5}  Created")
    click.echo("-" * 72)
    for exp in experiments:
        # Count runs
        runs = tracker.list_runs(experiment_id=exp["experiment_id"])
        click.echo(
            f"{exp['experiment_id']:<14} {exp['name']:<30} {len(runs):>5}  "
            f"{exp['created_at'][:19]}"
        )
    tracker.close()


@cli.command("runs")
@click.argument("experiment_id")
@click.option("--status", type=str, default=None, help="Filter by status.")
@click.pass_context
def list_runs(ctx: click.Context, experiment_id: str, status: Optional[str]) -> None:
    """List runs for an experiment."""
    tracker: ExperimentTracker = ctx.obj["tracker"]
    runs = tracker.list_runs(experiment_id=experiment_id, status=status)
    if not runs:
        click.echo("No runs found.")
        return
    click.echo(f"{'Run ID':<14} {'Name':<20} {'Status':<12} {'Duration':>10}  Started")
    click.echo("-" * 80)
    for run in runs:
        duration = (
            f"{run['duration_seconds']:.1f}s" if run["duration_seconds"] else "N/A"
        )
        name = run["run_name"] or run["run_id"][:8]
        click.echo(
            f"{run['run_id']:<14} {name:<20} {run['status']:<12} "
            f"{duration:>10}  {run['start_time'][:19]}"
        )
    tracker.close()


@cli.command("show")
@click.argument("run_id")
@click.pass_context
def show_run(ctx: click.Context, run_id: str) -> None:
    """Show details of a specific run."""
    tracker: ExperimentTracker = ctx.obj["tracker"]
    run = tracker.get_run(run_id)
    if run is None:
        click.echo(f"Run '{run_id}' not found.", err=True)
        sys.exit(1)
    click.echo(f"Run: {run['run_name'] or run['run_id']}")
    click.echo(f"  Status:   {run['status']}")
    click.echo(f"  Started:  {run['start_time']}")
    click.echo(f"  Ended:    {run['end_time'] or 'N/A'}")
    click.echo(f"  Duration: {run['duration_seconds']:.1f}s" if run["duration_seconds"] else "  Duration: N/A")
    if run["parameters"]:
        click.echo("\n  Parameters:")
        for k, v in run["parameters"].items():
            click.echo(f"    {k}: {v}")
    if run["metrics"]:
        click.echo("\n  Latest Metrics:")
        # Show unique latest metrics
        seen: dict[str, float] = {}
        for k, v in run["metrics"].items():
            seen[k] = v
        for k, v in sorted(seen.items()):
            click.echo(f"    {k}: {v}")
    if run["artifacts"]:
        click.echo("\n  Artifacts:")
        for a in run["artifacts"]:
            click.echo(f"    [{a['type']}] {a['name']} -> {a['path']}")
    tracker.close()


@cli.command("compare")
@click.argument("run_ids", nargs=-1, required=True)
@click.option("--metric", "-m", multiple=True, help="Metric keys to compare.")
@click.pass_context
def compare(
    ctx: click.Context, run_ids: tuple[str, ...], metric: tuple[str, ...]
) -> None:
    """Compare multiple runs side-by-side."""
    tracker: ExperimentTracker = ctx.obj["tracker"]
    metric_keys = list(metric) if metric else None
    comparison = tracker.compare_runs(list(run_ids), metric_keys)
    if not comparison:
        click.echo("No matching runs found.")
        return

    # Collect all param and metric keys
    all_params: set[str] = set()
    all_metrics: set[str] = set()
    for data in comparison.values():
        all_params.update(data.get("parameters", {}).keys())
        all_metrics.update(data.get("metrics", {}).keys())

    # Header
    ids = list(comparison.keys())
    col_width = 18
    header = f"{'Attribute':<22}"
    for rid in ids:
        name = comparison[rid].get("run_name") or rid[:8]
        header += f" {name:>{col_width}}"
    click.echo(header)
    click.echo("-" * len(header))

    # Status row
    row = f"{'status':<22}"
    for rid in ids:
        row += f" {comparison[rid]['status']:>{col_width}}"
    click.echo(row)

    # Parameter rows
    for pk in sorted(all_params):
        row = f"{'param.' + pk:<22}"
        for rid in ids:
            val = comparison[rid]["parameters"].get(pk, "N/A")
            row += f" {str(val):>{col_width}}"
        click.echo(row)

    # Metric rows
    for mk in sorted(all_metrics):
        row = f"{'metric.' + mk:<22}"
        for rid in ids:
            val = comparison[rid]["metrics"].get(mk)
            formatted = f"{val:.6f}" if val is not None else "N/A"
            row += f" {formatted:>{col_width}}"
        click.echo(row)

    tracker.close()


@cli.command("export")
@click.argument("output_path")
@click.option(
    "--format",
    "fmt",
    type=click.Choice(["csv", "json"]),
    default="csv",
    help="Export format.",
)
@click.option("--experiment", "experiment_id", default=None, help="Filter by experiment ID.")
@click.pass_context
def export_data(
    ctx: click.Context,
    output_path: str,
    fmt: str,
    experiment_id: Optional[str],
) -> None:
    """Export run data to CSV or JSON."""
    tracker: ExperimentTracker = ctx.obj["tracker"]
    if fmt == "csv":
        path = tracker.export_to_csv(output_path, experiment_id)
    else:
        path = tracker.export_to_json(output_path, experiment_id)
    click.echo(f"Exported to {path}")
    tracker.close()


@cli.command("history")
@click.argument("run_id")
@click.argument("metric_key")
@click.pass_context
def metric_history(ctx: click.Context, run_id: str, metric_key: str) -> None:
    """Show step-by-step metric history for a run."""
    tracker: ExperimentTracker = ctx.obj["tracker"]
    history = tracker.get_metric_history(run_id, metric_key)
    if not history:
        click.echo(f"No history found for metric '{metric_key}' in run '{run_id}'.")
        return
    click.echo(f"{'Step':>6}  {'Value':>12}")
    click.echo("-" * 22)
    for entry in history:
        click.echo(f"{entry['step']:>6}  {entry['value']:>12.6f}")
    tracker.close()


@cli.command("dashboard")
@click.option("--host", default="127.0.0.1", help="Dashboard host.")
@click.option("--port", default=5000, type=int, help="Dashboard port.")
@click.pass_context
def launch_dashboard(ctx: click.Context, host: str, port: int) -> None:
    """Launch the web dashboard."""
    tracker: ExperimentTracker = ctx.obj["tracker"]
    db_path = tracker.store.db_path
    tracker.close()

    from dashboard.app import create_app

    app = create_app(db_path)
    click.echo(f"Starting dashboard at http://{host}:{port}")
    app.run(host=host, port=port, debug=True)


def main() -> None:
    """Entry point for the CLI."""
    cli()


if __name__ == "__main__":
    main()
