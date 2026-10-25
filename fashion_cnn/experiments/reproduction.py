"""Run the complete recorded campaign in a new self-contained workspace."""

import shutil
from pathlib import Path

from ..analysis import run_analysis
from .stability import run_stability
from .evaluation import run_evaluation
from .explanation import explain
from .search import run_improvements
from .v1 import run_experiment


def reproduce(project_dir, destination):
    project_dir, destination = Path(project_dir).resolve(), Path(destination).resolve()
    data = project_dir / "data"
    if not (data / "FashionMNIST/raw/train-images-idx3-ubyte").is_file():
        raise FileNotFoundError("Download Fashion-MNIST first: python -m fashion_cnn download")
    if destination.is_relative_to(data):
        raise ValueError("The new workspace must be outside the source data directory.")
    destination.mkdir(parents=True, exist_ok=False)
    print(f"New workspace: {destination}", flush=True)
    shutil.copytree(data, destination / "data")
    results = destination / "results"
    results.mkdir()
    shutil.copy2(project_dir / "results/split_indices.npz", results / "split_indices.npz")
    run_experiment(destination, results / "experiment")
    run_analysis(destination, results / "analysis", include_final_test=True)
    run_improvements(destination, results / "improvement_v2")
    run_stability(destination, results / "stability_v2")
    run_evaluation(destination, results / "evaluation_v2", include_test=True)
    explain(destination, results / "explanation_v2")
    print(f"Reproduction complete: {results}", flush=True)
