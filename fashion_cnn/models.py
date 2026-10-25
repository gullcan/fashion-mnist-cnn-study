"""Model definitions shared by training, evaluation and browser export."""

from torch import nn


def build_cnn(channels=(6, 16), pooling="avg"):
    """Build the recorded CNN while preserving its Sequential state-dict keys.

    V1 uses (6, 16); the selected V2 uses (16, 32). Both have two
    convolution and two linear layers, and accept [N, 1, 28, 28] inputs.
    """
    if len(channels) != 2 or any(type(value) is not int or value < 1 for value in channels):
        raise ValueError("channels must contain two positive integers")
    if pooling not in ("avg", "max"):
        raise ValueError("pooling must be 'avg' or 'max'")
    first, second = channels
    pool = nn.AvgPool2d if pooling == "avg" else nn.MaxPool2d
    return nn.Sequential(
        nn.Conv2d(1, first, kernel_size=5),
        nn.ReLU(),
        pool(kernel_size=2, stride=2),
        nn.Conv2d(first, second, kernel_size=5),
        nn.ReLU(),
        pool(kernel_size=2, stride=2),
        nn.Flatten(start_dim=1),
        nn.Linear(second * 4 * 4, 84),
        nn.ReLU(),
        nn.Linear(84, 10),
    )


def build_baseline():
    """Flatten followed by one affine classifier: 7,850 parameters."""
    return nn.Sequential(nn.Flatten(start_dim=1), nn.Linear(28 * 28, 10))
