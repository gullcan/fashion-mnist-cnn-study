from torch import nn


def build_cnn():
    return nn.Sequential(
        nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=0),
        nn.ReLU(),
        nn.AvgPool2d(kernel_size=2, stride=2),

        nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),
        nn.ReLU(),
        nn.AvgPool2d(kernel_size=2, stride=2),

        nn.Flatten(start_dim=1),
        nn.Linear(in_features=16 * 4 * 4, out_features=84),
        nn.ReLU(),
        nn.Linear(in_features=84, out_features=10),
    )

def build_baseline():
    return nn.Sequential(
        nn.Flatten(start_dim=1),
        nn.Linear(in_features=28 * 28, out_features=10),
    )