from torch.utils.data.dataset import IterableDataset

from .base_dataset import BaseDataset


# ============================================
#                   QDataset
# ============================================
class QDataset(BaseDataset):
    """
    Basically a dataset version of the memory buffer. This object is
    used by lightning during training.
    """
    # -----
    # constructor
    # -----
    def __init__(self, memory, batchSize):
        self.memory = memory
        self.batchSize = batchSize

    # -----
    # iter
    # -----
    def __iter__(self):
        states, actions, rewards, nextStates, dones = self.memory.sample(self.batchSize)
        for i in range(len(dones)):
            yield states[i], actions[i], rewards[i], nextStates[i], dones[i]
