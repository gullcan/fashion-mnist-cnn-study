"""Training and evaluation loops, with image-weighted batch aggregation."""

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader


def train_one_epoch(model, loader, loss_fn, optimizer, device):
    model.train()

    loss_sum = 0.0
    correct_count = 0
    sample_count = 0

    for images, targets in loader:
        images = images.to(device)
        targets = targets.to(device)

        optimizer.zero_grad(set_to_none=True)

        logits = model(images)
        loss = loss_fn(logits, targets)

        loss.backward()
        optimizer.step()

        batch_count = len(targets)
        loss_sum += loss.item() * batch_count
        correct_count += (logits.argmax(dim=1) == targets).sum().item()
        sample_count += batch_count

    if sample_count == 0:
        raise ValueError("Cannot process an empty dataset.")
    mean_loss = loss_sum / sample_count
    accuracy = 100 * correct_count / sample_count

    return mean_loss, accuracy


def evaluate(model, loader, loss_fn, device):
    model.eval()

    loss_sum = 0.0
    correct_count = 0
    sample_count = 0

    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            targets = targets.to(device)

            logits = model(images)
            loss = loss_fn(logits, targets)

            batch_count = len(targets)
            loss_sum += loss.item() * batch_count
            correct_count += (logits.argmax(dim=1) == targets).sum().item()
            sample_count += batch_count

    if sample_count == 0:
        raise ValueError("Cannot process an empty dataset.")
    mean_loss = loss_sum / sample_count
    accuracy = 100 * correct_count / sample_count

    return mean_loss, accuracy


def collect_predictions(model, dataset, device, batch_size):
    """Her goruntuyu bir kez isle; agin ayarlarini degistirme."""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    model.eval()
    targets_all, predictions_all, probabilities_all = [], [], []
    loss_sum, sample_count = 0.0, 0
    loss_fn = nn.CrossEntropyLoss()
    with torch.no_grad():
        for images, targets in loader:
            images, targets = images.to(device), targets.to(device)
            logits = model(images)
            loss_sum += loss_fn(logits, targets).item() * len(targets)
            sample_count += len(targets)
            targets_all.append(targets.cpu().numpy())
            predictions_all.append(logits.argmax(dim=1).cpu().numpy())
            probabilities_all.append(logits.softmax(dim=1).cpu().numpy())
    if sample_count == 0:
        raise ValueError("Cannot process an empty dataset.")
    return (
        np.concatenate(targets_all), np.concatenate(predictions_all),
        np.concatenate(probabilities_all), loss_sum / sample_count,
    )


def synchronize(device):
    """GPU islemleri bitmeden sure olcumunu sonlandirma."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)
