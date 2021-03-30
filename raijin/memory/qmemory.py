from collections import deque
from typing import Tuple

import numpy as np
from omegaconf.dictconfig import DictConfig
import torch

from .base_memory import BaseMemory
from .experience import Experience


# ============================================
#                   QMemory
# ============================================
class QMemory(BaseMemory):
    """
    Implements the memory buffer from [Mnih et al. 2013][1].

    The buffer is a deque and sampling is done randomly and without
    replacement.

    [1]: https://arxiv.org/abs/1312.5602
    """

    __name__ = "QMemory"

    # -----
    # constructor
    # -----
    def __init__(self, params: DictConfig) -> None:
        self.capacity = params.capacity
        self.buffer = deque(maxlen=self.capacity)

    # -----
    # add
    # -----
    def add(self, experience: Experience) -> None:
        self.buffer.append(experience)

    # -----
    # sample
    # -----
    def sample(self, batchSize: int) -> Tuple:
        # Choose which experiences to grab
        indices = np.random.choice(len(self.buffer), batchSize, replace=False)
        # Extract those experiences from the buffer
        batch = zip(*[self.buffer[i] for i in indices])
        return self._process_batch(batch, batchSize)

    # -----
    # _process_batch
    # -----
    def _process_batch(self, batch, batchSize: int) -> Tuple:
        """
        Converts the components of the experiences within batch to
        tensors of the appropriate shape and type.
        """
        # Split the batch up into components. Each component is a tuple
        states, actions, rewards, nextStates, dones = batch
        # Each state and nextState is already a tensor, so we can just
        # stack them
        states = torch.stack(states)
        nextStates = torch.stack(nextStates)
        # Convert the other components to tensors
        actions = torch.from_numpy(np.array(actions))
        rewards = torch.from_numpy(np.array(rewards))
        dones = torch.from_numpy(np.array(dones))
        # Change dtype
        actions = actions.to(torch.float)
        rewards = rewards.to(torch.float)
        dones = dones.to(torch.float)
        # Reshape
        actions = actions.reshape((batchSize, 1))
        rewards = rewards.reshape((batchSize, 1))
        dones = dones.reshape((batchSize, 1))
        return (states, actions, rewards, nextStates, dones)

    # -----
    # state_dict
    # -----
    def state_dict(self) -> dict:
        return {}
