import torch
from torch import nn

from .base_network import BaseNetwork


# ============================================
#               DuelingQNetwork
# ============================================
class DuelingQNetwork(BaseNetwork):
    """
    Implements the network described in [Want et al. 2016][1].

    [1]: https://arxiv.org/abs/1511.06581
    """

    __name__ = "DuelingQNetwork"

    # -----
    # constructor
    # -----
    def __init__(self, inChannels: int, nActions: int, **kwargs: dict) -> None:
        super().__init__()
        # First convolutional layer
        conv1 = nn.Conv2d(
            in_channels=inChannels, out_channels=16, kernel_size=8, stride=4
        )
        conv1OutShape = self.get_conv_out_shape(110, 84, conv1)
        # Second convolutional layer
        conv2 = nn.Conv2d(
            in_channels=16, out_channels=32, kernel_size=4, stride=2
        )
        conv2OutShape = self.get_conv_out_shape(
            conv1OutShape[1], conv1OutShape[2], conv2
        )
        # Value stream
        valueLayer = nn.Linear(
            in_features=int(conv2OutShape.prod()), out_features=512
        )
        valueOutput = nn.Linear(in_features=512, out_features=1)
        # Advantage stream
        advantageLayer = nn.Linear(
            in_features=int(conv2OutShape.prod()), out_features=512
        )
        advantageOutput = nn.Linear(in_features=512, out_features=nActions)
        # Network components
        self.common = nn.Sequential(
            conv1,
            torch.nn.ReLU(),
            conv2,
            torch.nn.ReLU(),
            torch.nn.Flatten(),
        )
        self.valueStream = nn.Sequential(
            valueLayer, torch.nn.ReLU(), valueOutput
        )
        self.advantageStream = nn.Sequential(
            advantageLayer, torch.nn.ReLU(), advantageOutput
        )

    # -----
    # forward
    # -----
    def forward(self, x):
        common = self.common(x)
        value = self.valueStream(common)
        advantage = self.advantageStream(common)
        # Aggregate layer (eq. 9 in [1])
        return value + (advantage - torch.mean(advantage, 1, keepdim=True))
