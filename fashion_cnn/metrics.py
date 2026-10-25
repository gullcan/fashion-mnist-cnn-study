"""Classification metrics with explicit class order and zero-division policy."""

import numpy as np


def classification_metrics(targets, predictions, classes):
    """Precision=TP/tahmin, recall=TP/gercek, F1=2TP/(tahmin+gercek)."""
    count = len(classes)
    targets, predictions = np.asarray(targets), np.asarray(predictions)
    if count < 1 or targets.ndim != 1 or predictions.ndim != 1 or targets.shape != predictions.shape or targets.size == 0:
        raise ValueError("Metrics require matching nonempty one-dimensional label arrays.")
    for values in (targets, predictions):
        if not np.issubdtype(values.dtype, np.integer) or np.any(values < 0) or np.any(values >= count):
            raise ValueError("Labels must be integer class indices in range.")
    matrix = np.bincount(
        count * targets + predictions, minlength=count * count
    ).reshape(count, count)
    tp = matrix.diagonal().astype(np.float64)
    support = matrix.sum(axis=1)
    predicted_count = matrix.sum(axis=0)
    precision = np.divide(tp, predicted_count, out=np.zeros(count), where=predicted_count > 0)
    recall = np.divide(tp, support, out=np.zeros(count), where=support > 0)
    denominator = predicted_count + support
    f1 = np.divide(2 * tp, denominator, out=np.zeros(count), where=denominator > 0)
    per_class = [
        {
            "class_id": index, "class_name": name,
            "precision": float(precision[index]), "recall": float(recall[index]),
            "f1": float(f1[index]), "support": int(support[index]),
        }
        for index, name in enumerate(classes)
    ]
    metrics = {
        "samples": int(matrix.sum()), "correct": int(tp.sum()),
        "errors": int(matrix.sum() - tp.sum()),
        "accuracy_pct": float(100 * tp.sum() / matrix.sum()),
        "macro_f1": float(f1.mean()), "zero_division": 0,
    }
    return matrix, metrics, per_class
