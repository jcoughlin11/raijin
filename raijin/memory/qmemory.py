from collections import deque

from .base_memory import BaseMemory


# ============================================
#                   QMemory
# ============================================
class QMemory(BaseMemory):
    # -----
    # constructor
    # -----
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    # -----
    # len
    # -----
    def __len__(self):
        return len(self.buffer)

    # -----
    # capacity
    # -----
    @property
    def capacity(self):
        return self.buffer.maxlen

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
        # Recall that the experiences are tuples, so we build a list of tuples,
        # then pass those tuples as arguments to zip
        states, actions, rewards, nextStates, dones = zip(*[self.buffer[i] for i in indices])
        states = np.array(states)
        actions = np.array(actions)
        rewards = np.array(rewards, dtype=np.float)
        nextStates = np.array(nextStates)
        dones = np.array(dones, dtype=np.bool)
        return (states, actions, rewards, nextStates, dones)
