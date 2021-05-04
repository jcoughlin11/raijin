from typing import Tuple

import numpy as np
from omegaconf.dictconfig import DictConfig
import torch

from raijin.utilities.trees import SumTree

from .base_memory import BaseMemory
from .experience import Experience


# ============================================
#           PriorityMemory Class
# ============================================
class PriorityMemory(BaseMemory):
    """
    Memory buffer when prioritized experience replay is being used.
    Employs a sum tree instead of a deque.
    """

    __name__ = "PriorityMemory"

    # -----
    # Constructor
    # -----
    def __init__(self, params: DictConfig, device: str) -> None:
        self.capacity = params.capacity
        self.device = device
        self.buffer = SumTree(self.capacity)
        self.perA = params.perA
        self.perB = params.perB
        self.perBAnneal = params.perAnneal
        self.perE = params.perE
        self.upperPriority = 1.0
        self.indices = None

    # -----
    # Add
    # -----
    def add(self, experience: Experience) -> None:
        """
        Stores the experience, along with a priority, to the buffer.
        According to Schaul16 algorithm 1, the new experiences are
        added with a priority equal to the tree's current max priority.
        """
        # Get the current max priority in the tree. Priorities are
        # held in the leaf nodes
        maxPriority = np.max(self.buffer.tree[-self.buffer.nLeafs :])
        # If maxPriority is 0, use upperPriority because 0 means the
        # experience will never be chosen
        if maxPriority == 0:
            maxPriority = self.upperPriority
        self.buffer.add(experience, maxPriority)

    # -----
    # Sample
    # -----
    def sample(self, batchSize: int) -> Tuple:
        """
        The probability for a particular experience to be chosen is
        given by equation 1 in Schaul16. The details of how to sample
        from the sumtree are given in Appendix B.2.1: Proportional
        prioritization in Schaul16.

        Essentially, we break up the range [0, priority_total] into
        batchSize segments of equal size. We then uniformly choose a
        value from each segment and get the experiences that correspond
        to each of these sampled values.
        """
        # We need to return the selected samples (to be used in
        # training), the indices of these samples (so that the tree can
        # be properly updated), and the importance sampling weights to
        # be used in training
        self.indices = np.zeros((batchSize,), dtype=np.int32)
        priorities = np.zeros((batchSize, 1))
        experiences = []
        # We need to break up the range [0, p_tot] equally into
        # batchSize segments, so here we get the width of each segment
        segmentWidth = self.buffer.total_priority / batchSize
        # Anneal the strength of the IS weights (cap the parameter at 1)
        self.perB = np.min([1.0, self.perB + self.perBAnneal])
        # Loop over the desired number of samples
        for i in range(batchSize):
            # We need to uniformly select a value from each segment, so
            # here we get the lower and upper bounds of the segment
            lowerBound = i * segmentWidth
            upperBound = (i + 1) * segmentWidth
            # Choose a value from within the segment
            value = np.random.uniform(lowerBound, upperBound)
            # Retrieve the experience whose priority matches value from
            # the tree
            index, priority, experience = self.buffer.get_leaf(value)
            self.indices[i] = index
            priorities[i, 0] = priority
            experiences.append(experience)
        # Calculate the importance sampling weights (I think it's (N,1)
        samplingProbabilities = priorities / self.buffer.total_priority
        isWeights = np.power(batchSize * samplingProbabilities, -self.perB)
        isWeights = isWeights / np.max(isWeights)
        batch = zip(*[e for e in experiences])
        return self._process_batch(batch, batchSize, isWeights)

    # -----
    # _process_batch
    # -----
    def _process_batch(
        self, batch, batchSize: int, isWeights: np.ndarray
    ) -> Tuple:
        # Split the batch up into components. Each component is a tuple
        states, actions, rewards, nextStates, dones = batch
        # Each state and nextState is already a tensor, so we can just
        # stack them. Their shape should be (N, C, H, W)
        states = torch.stack(states)
        nextStates = torch.stack(nextStates)
        # Convert the other components to tensors
        actions = torch.from_numpy(np.array(actions))
        rewards = torch.from_numpy(np.array(rewards))
        dones = torch.from_numpy(np.array(dones))
        isWeights = torch.from_numpy(isWeights)
        # Change dtype
        actions = actions.to(torch.float)
        rewards = rewards.to(torch.float)
        dones = dones.to(torch.float)
        isWeights = isWeights.to(torch.float)
        # Reshape
        actions = actions.reshape((batchSize, 1))
        rewards = rewards.reshape((batchSize, 1))
        dones = dones.reshape((batchSize, 1))
        isWeights = isWeights.reshape((batchSize, 1))
        # Move to desired device
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        nextStates = nextStates.to(self.device)
        dones = dones.to(self.device)
        isWeights = isWeights.to(self.device)
        return (states, actions, rewards, nextStates, dones, isWeights)

    # -----
    # Update
    # -----
    def update(self, absErrors: torch.Tensor) -> None:
        """
        This function uses the new errors generated from training in
        order to update the priorities for those experiences that were
        selected in sample().
        """
        # Calculate priorities from errors (proportional prioritization)
        priorities = absErrors + self.perE
        # Clip the errors for stability
        priorities = torch.minimum(priorities, torch.tensor(self.upperPriority))
        # Apply alpha
        priorities = torch.pow(priorities, self.perA)
        # Update the tree. self.indices should be a numpy array, so don't
        # need .item()
        for ind, p in zip(self.indices, priorities):
            self.buffer.update(ind, p.item())

    # -----
    # state_dict
    # -----
    def state_dict(self) -> dict:
        return {}
