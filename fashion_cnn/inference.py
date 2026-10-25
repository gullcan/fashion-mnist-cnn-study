"""Saved-model inference, independent of training workflows and plotting."""

import json
from pathlib import Path

import torch

from .artifacts import file_hash
from .data import load_split
from .models import build_cnn


def load_checkpoint(path, expected_hash=None):
    """Load a CNN training artifact on CPU without constructing an optimizer."""
    path = Path(path)
    if expected_hash is not None and file_hash(path) != expected_hash:
        raise ValueError("Checkpoint does not match the evaluated model.")
    saved = torch.load(path, map_location="cpu", weights_only=True)
    config, classes = saved["candidate_config"], saved["classes"]
    if not isinstance(classes, list) or len(classes) != 10 or len(set(classes)) != 10:
        raise ValueError("Checkpoint must define the ten distinct class names.")
    model = build_cnn(channels=config["channels"], pooling=config["pooling"])
    model.load_state_dict(saved["model_state_dict"])
    model.eval()
    return model, classes


def load_selected_model(project_dir):
    """Load the published seed42 artifact and enforce its frozen identity."""
    root = Path(project_dir)
    report = json.loads((root / "results/improvement_v2/summary.json").read_text(encoding="utf-8"))
    frozen = json.loads((root / "results/evaluation_v2/frozen_protocol.json").read_text(encoding="utf-8"))
    checkpoint = root / "results/improvement_v2" / report["winner"]["checkpoint"]
    return load_checkpoint(checkpoint, expected_hash=frozen["identity"]["checkpoint_sha256"])


def predict_tensor(model, image, classes):
    """Classify one float32 [1,28,28] image without a confidence claim."""
    if not isinstance(image, torch.Tensor) or image.shape != (1, 28, 28) or image.dtype != torch.float32:
        raise ValueError("Expected one float32 grayscale tensor with shape [1,28,28].")
    if not torch.isfinite(image).all() or image.min() < 0 or image.max() > 1:
        raise ValueError("Pixels must be finite values in [0,1].")
    model.eval()
    with torch.no_grad():
        probabilities = model(image.unsqueeze(0).to(next(model.parameters()).device)).softmax(dim=1)[0]
    if len(probabilities) != len(classes):
        raise ValueError("Class names do not match the model output.")
    scores, labels = probabilities.topk(3)
    return {
        "predicted_class": classes[int(labels[0])],
        "top3": [{"class": classes[int(label)], "softmax_score": float(score)}
                 for label, score in zip(labels, scores)],
        "score_note": "Softmax scores are not calibrated correctness guarantees.",
    }


def predict_sample(project_dir, index, checkpoint=None):
    """Inspect one source image; download and test evaluation stay explicit."""
    from torchvision.datasets import FashionMNIST
    from torchvision.transforms import ToTensor

    root = Path(project_dir)
    if not 0 <= index < 60000:
        raise ValueError("Index must be between 0 and 59999.")
    model, classes = load_selected_model(root) if checkpoint is None else load_checkpoint(checkpoint)
    dataset = FashionMNIST(root / "data", train=True, download=False, transform=ToTensor())
    if list(dataset.classes) != classes:
        raise ValueError("Dataset label order differs from the checkpoint.")
    image, target = dataset[index]
    _, validation, _ = load_split(root)
    return {
        "source_index": index,
        "split": "validation" if (validation == index).any() else "training",
        "true_class": classes[target],
        **predict_tensor(model, image, classes),
    }
