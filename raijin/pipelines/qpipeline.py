from collections import deque
from typing import Deque

import numpy as np
from omegaconf.dictconfig import DictConfig
import torch
import torchvision.transforms.functional as tf

from .base_pipeline import BasePipeline


# ============================================
#                  QPipeline
# ============================================
class QPipeline(BasePipeline):
    __name__ = "QPipeline"

    # -----
    # constructor
    # -----
    def __init__(self, params: DictConfig) -> None:
        self.normValue = params.normValue
        self.traceLen = params.traceLen
        self.offsetHeight = params.offsetHeight
        self.offsetWidth = params.offsetWidth
        self.cropHeight = params.cropHeight
        self.cropWidth = params.cropWidth
        self.frameStack: Deque[torch.Tensor] = deque(maxlen=self.traceLen)

    # -----
    # normalize_frame
    # -----
    def normalize_frame(self, frame: torch.Tensor) -> torch.Tensor:
        """
        Rescales each rgb value to be between 0 and 1.

        torchvision's normalize function isn't used because that
        computes x' = (x - mu) / sigma. Since we don't have all of the
        data beforehand, getting mu and sigma is problematic.
        """
        return frame / self.normValue

    # -----
    # grayscale
    # -----
    def grayscale(self, frame: torch.Tensor) -> torch.Tensor:
        """
        Converts the image to have just one channel.

        The shape of the image must be (..., H, W) where ... denotes an
        arbitrary number of dimensions.
        """
        return tf.rgb_to_grayscale(frame)

    # -----
    # crop
    # -----
    def crop(self, frame: torch.Tensor) -> torch.Tensor:
        """
        Cuts out the unnecessary parts of the image.

        The shape of the image must be (..., H, W) where ... denotes an
        arbitrary number of dimensions.
        """
        return tf.crop(
            frame,
            self.offsetHeight,
            self.offsetWidth,
            self.cropHeight,
            self.cropWidth,
        )

    # -----
    # stack
    # -----
    def stack(self, frame: torch.Tensor, newEpisode: bool) -> torch.Tensor:
        """
        Adds the given frame to the top of a pile containing the
        preceding frames.

        This is done to help with the problem of motion.
        """
        # The channel dimension isn't needed
        frame = torch.squeeze(frame)
        if newEpisode:
            for _ in range(self.traceLen):
                self.frameStack.append(frame)
        else:
            self.frameStack.append(frame)
        # torch tensors are always (N, C, H, W) so we can use axis=0
        return torch.stack(list(self.frameStack))

    # -----
    # process
    # -----
    def process(self, frame: np.ndarray, newEpisode: bool) -> torch.Tensor:
        frame = self._reshape_frame(frame)
        frameTensor = torch.from_numpy(frame)
        normFrame = self.normalize_frame(frameTensor)
        grayFrame = self.grayscale(normFrame)
        cropFrame = self.crop(grayFrame)
        state = self.stack(cropFrame, newEpisode)
        return state

    # -----
    # state_dict
    # -----
    def state_dict(self) -> dict:
        return {"frameStack": self.frameStack}
