from abc import ABC
from abc import abstractmethod

import numpy as np
import torch

from raijin.utilities.register import register_object


# ============================================
#               BasePipeline
# ============================================
class BasePipeline(ABC):
    __name__ = "BasePipeline"

    # -----
    # subclass_hook
    # -----
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        register_object(cls)

    # -----
    # processs
    # -----
    @abstractmethod
    def process(self, frame: np.ndarray, newEpisode: bool) -> torch.Tensor:
        """
        Performs any desired image processing on the given game frame.
        """
        pass

    # -----
    # state_dict
    # -----
    @abstractmethod
    def state_dict(self) -> dict:
        pass

    # -----
    # _reshape_frame
    # -----
    def _reshape_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        The environment produces an array of shape (H, W, C), but
        pytorch needs the channels to be first.
        """
        return frame.reshape(
            [
                frame.shape[-1],
            ]
            + list(frame.shape[:-1])
        )
