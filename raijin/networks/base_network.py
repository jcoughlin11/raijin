from abc import ABC
from abc import abstractmethod

import numpy as np
import torch

from raijin.utilities.register import register_object


# ============================================
#                 BaseNetwork
# ============================================
class BaseNetwork(ABC, torch.nn.Module):
    __name__ = "BaseNetwork"

    # -----
    # subclass_hook
    # -----
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        register_object(cls)

    # -----
    # constructor
    # -----
    def __init__(self) -> None:
        # I think this works with multiple parents? Check.
        super().__init__()

    # -----
    # forward
    # -----
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines a forward pass through the network.
        """
        pass

    # -----
    # get_conv_out_shape
    # -----
    def get_conv_out_shape(
        self, H_in: int, W_in: int, convLayer: torch.nn.Conv2d
    ) -> np.ndarray:
        """
        Calculates the shape of the tensor output by the given convolutional
        layer. Does NOT include the batch dimension.
        See: https://tinyurl.com/3h6cm9fk
        """
        pad = convLayer.padding
        dil = convLayer.dilation
        ks = convLayer.kernel_size
        s = convLayer.stride
        h = H_in
        w = W_in
        # These two calcs can be vectorized
        H_out = (h + 2 * pad[0] - dil[0] * (ks[0] - 1) - 1) / s[0] + 1
        W_out = (w + 2 * pad[1] - dil[1] * (ks[1] - 1) - 1) / s[1] + 1
        return np.array([convLayer.out_channels, H_out, W_out], dtype=np.int32)
