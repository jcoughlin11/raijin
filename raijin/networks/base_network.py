from abc import ABC
from abc import abstractmethod

import numpy as np
from torch import nn

from raijin.utilities.register import register_object


# ============================================
#                 BaseNetwork
# ============================================
class BaseNetwork(ABC, nn.Module):
    # -----
    # subclass hook
    # -----
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        register_object(cls)

    # -----
    # constructor
    # -----
    def __init__(self):
        # Not sure about this, since this class has two parents
        super().__init__()

    # -----
    # forward
    # -----
    @abstractmethod
    def forward(self):
        pass

    # -----
    # get_conv_out_shape
    # -----
    def get_conv_out_shape(self, H_in, W_in, convLayer):
        """
        Calculates the shape of the tensor output by the given convolutional
        layer. Does NOT include the batch dimension. See:
        https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html#torch.nn.Conv2d
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
        return np.array([convLayer.out_channels, H_out, W_out])
