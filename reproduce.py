"""Rerun the recorded study in a fresh workspace; keep original artifacts intact."""

import argparse
import shutil
from pathlib import Path

from analyze_experiment import run_analysis
from check_v2_stability import run_stability
from evaluate_v2 import run_evaluation
from explain_v2 import explain
from improve_cnn import run_improvements
from run_experiment import run_experiment


def reproduce(project_dir, destination):
    project_dir, destination = Path(project_dir).resolve(), Path(destination).resolve()
    data = project_dir / "data"
    if not (data / "FashionMNIST/raw/train-images-idx3-ubyte").is_file():
        raise FileNotFoundError("Download Fashion-MNIST first: python inspect_dataset.py")
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True,
                        help="New directory that does not already exist.")
    args = parser.parse_args()
    reproduce(Path(__file__).resolve().parent, args.output)

