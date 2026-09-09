# ML Experiment Tracker - Lightweight Experiment Management
# Author: Maharshi Soni | License: MIT
"""
Synthetic data generator for demonstration and testing.
Creates realistic ML experiment data with multiple training runs,
varying hyperparameters, and realistic loss/accuracy curves.
"""

from __future__ import annotations

import math
import os
import random
from typing import Dict, List, Tuple

from tracker.tracker import ExperimentTracker


def _generate_loss_curve(
    epochs: int,
    initial_loss: float,
    final_loss: float,
    noise: float = 0.02,
    seed: int = 42,
) -> List[float]:
    """Generate a realistic exponential-decay loss curve with noise."""
    rng = random.Random(seed)
    decay_rate = -math.log(final_loss / initial_loss) / epochs
    values = []
    for epoch in range(epochs):
        base = initial_loss * math.exp(-decay_rate * epoch)
        noisy = base + rng.gauss(0, noise * base)
        values.append(max(noisy, final_loss * 0.5))
    return values


def _generate_accuracy_curve(
    epochs: int,
    initial_acc: float,
    final_acc: float,
    noise: float = 0.01,
    seed: int = 42,
) -> List[float]:
    """Generate a realistic accuracy curve that approaches a ceiling."""
    rng = random.Random(seed)
    values = []
    for epoch in range(epochs):
        t = epoch / max(epochs - 1, 1)
        base = initial_acc + (final_acc - initial_acc) * (1 - math.exp(-3 * t))
        noisy = base + rng.gauss(0, noise)
        values.append(min(max(noisy, 0.0), 1.0))
    return values


def generate_cnn_experiment(tracker: ExperimentTracker) -> str:
    """Generate a CNN image classification experiment with multiple runs."""
    exp_id = tracker.create_experiment(
        name="mnist-cnn",
        description="Convolutional Neural Network experiments on MNIST digit classification",
        tags={"dataset": "mnist", "task": "classification", "framework": "pytorch"},
    )

    configs = [
        {
            "run_name": "baseline-small-lr",
            "params": {
                "learning_rate": 0.0001,
                "batch_size": 32,
                "optimizer": "adam",
                "num_layers": 3,
                "hidden_dim": 64,
                "dropout": 0.2,
                "epochs": 20,
            },
            "final_loss": 0.25,
            "final_acc": 0.92,
        },
        {
            "run_name": "medium-lr",
            "params": {
                "learning_rate": 0.001,
                "batch_size": 64,
                "optimizer": "adam",
                "num_layers": 3,
                "hidden_dim": 128,
                "dropout": 0.3,
                "epochs": 20,
            },
            "final_loss": 0.08,
            "final_acc": 0.97,
        },
        {
            "run_name": "large-lr-sgd",
            "params": {
                "learning_rate": 0.01,
                "batch_size": 128,
                "optimizer": "sgd",
                "num_layers": 4,
                "hidden_dim": 256,
                "dropout": 0.4,
                "epochs": 20,
            },
            "final_loss": 0.12,
            "final_acc": 0.955,
        },
        {
            "run_name": "high-dropout",
            "params": {
                "learning_rate": 0.001,
                "batch_size": 64,
                "optimizer": "adam",
                "num_layers": 3,
                "hidden_dim": 128,
                "dropout": 0.6,
                "epochs": 20,
            },
            "final_loss": 0.18,
            "final_acc": 0.94,
        },
        {
            "run_name": "deep-network",
            "params": {
                "learning_rate": 0.0005,
                "batch_size": 32,
                "optimizer": "adamw",
                "num_layers": 6,
                "hidden_dim": 256,
                "dropout": 0.3,
                "epochs": 20,
            },
            "final_loss": 0.06,
            "final_acc": 0.982,
        },
    ]

    for i, config in enumerate(configs):
        with tracker.start_run(
            exp_id,
            run_name=config["run_name"],
            tags={"model_type": "cnn", "trial": str(i + 1)},
        ) as run_id:
            tracker.log_params(run_id, config["params"])
            epochs = config["params"]["epochs"]
            losses = _generate_loss_curve(
                epochs, 2.5, config["final_loss"], seed=i * 100
            )
            accuracies = _generate_accuracy_curve(
                epochs, 0.1, config["final_acc"], seed=i * 100 + 1
            )
            val_losses = _generate_loss_curve(
                epochs,
                2.5,
                config["final_loss"] * 1.2,
                noise=0.04,
                seed=i * 100 + 2,
            )
            val_accuracies = _generate_accuracy_curve(
                epochs,
                0.1,
                config["final_acc"] * 0.98,
                noise=0.015,
                seed=i * 100 + 3,
            )
            for epoch in range(epochs):
                tracker.log_metrics(
                    run_id,
                    {
                        "train_loss": losses[epoch],
                        "train_accuracy": accuracies[epoch],
                        "val_loss": val_losses[epoch],
                        "val_accuracy": val_accuracies[epoch],
                    },
                    step=epoch,
                )
            tracker.log_artifact(
                run_id,
                name=f"model_{config['run_name']}.pt",
                path=f"artifacts/models/model_{config['run_name']}.pt",
                artifact_type="model",
                size_bytes=random.randint(500_000, 5_000_000),
                metadata={"format": "pytorch", "framework_version": "2.1.0"},
            )
    return exp_id


def generate_nlp_experiment(tracker: ExperimentTracker) -> str:
    """Generate an NLP text classification experiment."""
    exp_id = tracker.create_experiment(
        name="sentiment-analysis",
        description="Sentiment analysis on IMDB reviews with transformer models",
        tags={"dataset": "imdb", "task": "sentiment", "framework": "pytorch"},
    )

    configs = [
        {
            "run_name": "bert-base",
            "params": {
                "model": "bert-base-uncased",
                "learning_rate": 2e-5,
                "batch_size": 16,
                "max_length": 256,
                "warmup_steps": 500,
                "epochs": 5,
            },
            "final_loss": 0.22,
            "final_acc": 0.91,
        },
        {
            "run_name": "bert-large",
            "params": {
                "model": "bert-large-uncased",
                "learning_rate": 1e-5,
                "batch_size": 8,
                "max_length": 512,
                "warmup_steps": 1000,
                "epochs": 5,
            },
            "final_loss": 0.15,
            "final_acc": 0.94,
        },
        {
            "run_name": "distilbert",
            "params": {
                "model": "distilbert-base-uncased",
                "learning_rate": 5e-5,
                "batch_size": 32,
                "max_length": 256,
                "warmup_steps": 200,
                "epochs": 5,
            },
            "final_loss": 0.30,
            "final_acc": 0.88,
        },
    ]

    for i, config in enumerate(configs):
        with tracker.start_run(
            exp_id,
            run_name=config["run_name"],
            tags={"model_type": "transformer", "trial": str(i + 1)},
        ) as run_id:
            tracker.log_params(run_id, config["params"])
            epochs = config["params"]["epochs"]
            losses = _generate_loss_curve(
                epochs, 1.5, config["final_loss"], noise=0.03, seed=i * 200
            )
            accuracies = _generate_accuracy_curve(
                epochs, 0.5, config["final_acc"], noise=0.02, seed=i * 200 + 1
            )
            f1_scores = _generate_accuracy_curve(
                epochs, 0.45, config["final_acc"] - 0.01, noise=0.02, seed=i * 200 + 2
            )
            for epoch in range(epochs):
                tracker.log_metrics(
                    run_id,
                    {
                        "loss": losses[epoch],
                        "accuracy": accuracies[epoch],
                        "f1_score": f1_scores[epoch],
                    },
                    step=epoch,
                )
            tracker.log_artifact(
                run_id,
                name=f"model_{config['run_name']}.bin",
                path=f"artifacts/models/model_{config['run_name']}.bin",
                artifact_type="model",
                size_bytes=random.randint(100_000_000, 500_000_000),
                metadata={"format": "huggingface", "task": "text-classification"},
            )
    return exp_id


def generate_regression_experiment(tracker: ExperimentTracker) -> str:
    """Generate a regression experiment comparing model types."""
    exp_id = tracker.create_experiment(
        name="house-price-regression",
        description="House price prediction with different regression models",
        tags={"dataset": "california_housing", "task": "regression", "framework": "sklearn"},
    )

    configs = [
        {
            "run_name": "linear-regression",
            "params": {
                "model_type": "linear",
                "features": "all",
                "normalize": "true",
                "alpha": 0.0,
            },
            "final_mse": 0.52,
            "final_r2": 0.61,
        },
        {
            "run_name": "ridge-alpha-1",
            "params": {
                "model_type": "ridge",
                "features": "all",
                "normalize": "true",
                "alpha": 1.0,
            },
            "final_mse": 0.48,
            "final_r2": 0.65,
        },
        {
            "run_name": "random-forest",
            "params": {
                "model_type": "random_forest",
                "n_estimators": 100,
                "max_depth": 15,
                "min_samples_split": 5,
            },
            "final_mse": 0.25,
            "final_r2": 0.82,
        },
        {
            "run_name": "gradient-boosting",
            "params": {
                "model_type": "gradient_boosting",
                "n_estimators": 200,
                "max_depth": 6,
                "learning_rate": 0.1,
            },
            "final_mse": 0.19,
            "final_r2": 0.87,
        },
    ]

    rng = random.Random(42)
    for i, config in enumerate(configs):
        with tracker.start_run(
            exp_id,
            run_name=config["run_name"],
            tags={"model_type": config["params"].get("model_type", "unknown")},
        ) as run_id:
            tracker.log_params(run_id, config["params"])
            # For non-iterative models, log single-step metrics
            tracker.log_metrics(
                run_id,
                {
                    "mse": config["final_mse"] + rng.gauss(0, 0.01),
                    "rmse": (config["final_mse"] + rng.gauss(0, 0.01)) ** 0.5,
                    "r2_score": config["final_r2"] + rng.gauss(0, 0.005),
                    "mae": config["final_mse"] * 0.7 + rng.gauss(0, 0.01),
                },
                step=0,
            )
    return exp_id


def main() -> None:
    """Generate all synthetic data."""
    db_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "experiments.db"
    )
    # Remove existing database for a clean start
    if os.path.exists(db_path):
        os.remove(db_path)

    tracker = ExperimentTracker(db_path)
    print("Generating synthetic experiment data...")

    exp1 = generate_cnn_experiment(tracker)
    print(f"  Created CNN experiment: {exp1} (5 runs, 20 epochs each)")

    exp2 = generate_nlp_experiment(tracker)
    print(f"  Created NLP experiment: {exp2} (3 runs, 5 epochs each)")

    exp3 = generate_regression_experiment(tracker)
    print(f"  Created regression experiment: {exp3} (4 runs)")

    tracker.close()
    print(f"\nDone! Database saved to: {db_path}")
    print(f"Total: 3 experiments, 12 runs")
    print(f"\nLaunch the dashboard with:")
    print(f"  python cli.py --db {db_path} dashboard")


if __name__ == "__main__":
    main()
