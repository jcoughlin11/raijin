from collections import deque

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
    def __init__(self, params):
        self.normValue = params.normValue
        self.traceLen = params.traceLen
        self.offsetHeight = params.offsetHeight
        self.offsetWidth = params.offsetWidth
        self.cropHeight = params.cropHeight
        self.cropWidth = params.cropWidth
        self.frameStack = deque(maxlen=self.traceLen)

    # -----
    # normalize_frame
    # -----
    def normalize_frame(self, frame):
        """
        I don't use tf.normalize here because that does x' = (x - mu)/sigma
        and since I don't have all of the data beforehand, I'm not sure
        what to do about mu and sigma. Doing it on a frame-by-frame basis
        seems wrong.
        """
        return frame / self.normValue

    # -----
    # grayscale
    # -----
    def grayscale(self, frame):
        # Shape must be (..., H, W) where ... denotes an arbitrary number
        # of dimensions
        return tf.rgb_to_grayscale(frame)

    # -----
    # crop
    # -----
    def crop(self, frame):
        # Shape must be (..., H, W) where ... denotes an arbitrary number
        # of dimensions
        return tf.crop(
            frame,
            self.offsetHeight,
            self.offsetWidth,
            self.cropHeight,
            self.cropWidth
        )

    # -----
    # stack
    # -----
    def stack(self, frame, newEpisode):
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
    def process(self, frame, newEpisode):
        frame = torch.from_numpy(frame)
        frame = self.normalize_frame(frame)
        frame = self.grayscale(frame)
        frame = self.crop(frame)
        state = self.stack(frame, newEpisode)
        return state

    # -----
    # state_dict
    # -----
    def state_dict(self):
        return {"frameStack" : self.frameStack}
