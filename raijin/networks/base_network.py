from abc import ABC
from abc import abstractmethod

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
