from torch import nn

from .base_network import BaseNetwork


# ============================================
#                  QNetwork
# ============================================
class QNetwork(BaseNetwork):
    __name__ = "QNetwork"
    # -----
    # constructor
    # -----
    def __init__(self, inChannels, nActions, **kwargs):
        super().__init__()
        # First convolutional layer
        conv1 = nn.Conv2d(
            in_channels=inChannels,
            out_channels=16,
            kernel_size=8,
            stride=4
        )
        conv1OutShape = self.get_conv_out_shape(110, 84, conv1)
        # Second convolutional layer
        conv2 = nn.Conv2d(
            in_channels=16,
            out_channels=32,
            kernel_size=4,
            stride=2
        )
        conv2OutShape = self.get_conv_out_shape(conv1OutShape[1], conv1OutShape[2], conv2)
        # First fully connected layer (torch's Linear -> tf's Dense)
        fc1 = nn.Linear(in_features=conv2OutShape.prod(), out_features=256)
        # Output layer
        outputLayer = nn.Linear(in_features=256, out_features=nActions)
        # Network. Note that the last linear activation function isn't
        # needed here as it was in tf. In tf we had to specify an
        # activation, and linear is just g(x) = x
        self.net = nn.Sequential(
            conv1,
            nn.ReLU(),
            conv2,
            nn.ReLU(),
            nn.Flatten(),
            fc1,
            nn.ReLU(),
            outputLayer
        )

    # -----
    # forward
    # -----
    def forward(self, x):
        return self.net(x)
