from collections import deque

import numpy as np
import torch

from .base_memory import BaseMemory


# ============================================
#                   QMemory
# ============================================
class QMemory(BaseMemory):
    # -----
    # constructor
    # -----
    def __init__(self, params):
        self.capacity = params.capacity
        self.buffer = deque(maxlen=self.capacity)

    # -----
    # add
    # -----
    def add(self, experience):
        self.buffer.append(experience)

    # -----
    # sample
    # -----
    def sample(self, batchSize):
        indices = np.random.choice(len(self.buffer), batchSize, replace=False) 
        batch = zip(*[self.buffer[i] for i in indices])
        states, actions, rewards, nextStates, dones = batch
        # states and nextStates should already be tensors. The others need to
        # be converted to one
        # Convert to tensors
        actions = torch.from_numpy(actions)
        rewards = torch.from_numpy(rewards)
        dones = torch.from_numpy(dones)
        # Change dtype
        actions = actions.to(torch.float)
        rewards = rewards.to(torch.float)
        dones = torch.to(torch.float)
        # Shapes
        # states: (batchSize, traceLen, cropHeight, cropWidth)
        # nextStates: (batchSize, traceLen, cropHeight, cropWidth)
        # actions: (batchSize, 1)
        # rewards: (batchSize, 1)
        # dones: (batchSize, 1)
        actions = actions.reshape((batchSize, 1))
        rewards = rewards.reshape((batchSize, 1))
        dones = dones.reshape((batchSize, 1))
        return (states, actions, rewards, nextStates, dones)
