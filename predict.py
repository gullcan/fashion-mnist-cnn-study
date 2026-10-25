"""Run the selected checkpoint on one training-source image without training."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torchvision.datasets import FashionMNIST
from torchvision.transforms import ToTensor

from analyze_experiment import file_hash
from improve_cnn import build_candidate


def load_selected_model(project_dir):
    project_dir = Path(project_dir)
    report = json.loads((project_dir / "results/improvement_v2/summary.json").read_text(encoding="utf-8"))
    frozen = json.loads((project_dir / "results/evaluation_v2/frozen_protocol.json").read_text(encoding="utf-8"))
    checkpoint = project_dir / "results/improvement_v2" / report["winner"]["checkpoint"]
    if file_hash(checkpoint) != frozen["identity"]["checkpoint_sha256"]:
        raise ValueError("Checkpoint does not match the evaluated model.")
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model = build_candidate(saved["candidate_config"])
    model.load_state_dict(saved["model_state_dict"])
    model.eval()
    return model, saved["classes"]


def predict_sample(project_dir, index):
    project_dir = Path(project_dir)
    model, classes = load_selected_model(project_dir)
    dataset = FashionMNIST(project_dir / "data", train=True, download=False, transform=ToTensor())
    if not 0 <= index < len(dataset):
        raise ValueError(f"Index must be between 0 and {len(dataset) - 1}.")
    if list(dataset.classes) != classes:
        raise ValueError("Dataset label order differs from the checkpoint.")
    image, target = dataset[index]
    with torch.no_grad():
        probabilities = model(image.unsqueeze(0)).softmax(dim=1)[0]
    scores, labels = probabilities.topk(3)
    with np.load(project_dir / "results/split_indices.npz", allow_pickle=False) as split:
        part = "validation" if np.any(split["val_indices"] == index) else "training"
    return {
        "source_index": index, "split": part, "true_class": classes[target],
        "predicted_class": classes[int(labels[0])],
        "top3": [{"class": classes[int(label)], "softmax_score": float(score)}
                 for label, score in zip(labels, scores)],
        "score_note": "Softmax scores are not calibrated correctness guarantees.",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=int, default=49804)
    args = parser.parse_args()
    print(json.dumps(predict_sample(Path(__file__).resolve().parent, args.index), indent=2))

