from collections import deque

import numpy as np

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
        return zip(*[self.buffer[i] for i in indices])
