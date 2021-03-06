import numpy as np
import torch

from .base_agent import BaseAgent
from .experience import Experience


# ============================================
#                   QAgent
# ============================================
class QAgent(BaseAgent):
    """
    Epsilon-greedy action selection and environment interaction as
    described in Mnih et al. 2013.
    """
    # -----
    # constructor
    # -----
    def __init__(self, env, memory, actionParams):
        self.env = env
        self.memory = memory
        self.epsilonStart = actionParams.epsilonStart
        self.epsilonStop = actionParams.epsilonStop
        self.epsilonDecayRate = actionParams.epsilonDecayRate
        self.decayStep = 0
        self.state = self.env.reset()

    # -----
    # reset
    # -----
    def reset(self):
        self.state = self.env.reset()

    # -----
    # choose_action
    # -----
    def choose_action(self, net):
        exploitProb = np.random.random()
        exploreProb = self.epsilonStop + (
            self.epsilonStart - self.epsilonStop
        ) * np.exp(-self.epsilonDecayRate * self.decayStep)
        self.decayStep += 1
        if exploreProb >= exploitProb:
            action = self.env.action_space.sample()
        else:
            # The [] here is so that a batch dimension of size 1 is added
            # This can also be done with state.unsqueeze(0) after state has
            # been converted to a tensor via state = torch.tensor(self.state)
            state = torch.tensor([self.state])
            # qValues is a tensor of shape (1, nActions)
            qValues = net(state)
            # torch.max returns two things: values, indices. values is
            # a tensor of the actual max value along the specified dimension
            # and indices is a tensor of the index value of the max value
            # along the specified dimension. Recall that dim=0 operates along
            # the columns and dim=1 operates along the row. We don't need
            # the actual max value, just the index (since the index corresponds
            # to the action to take)
            _, indices = torch.max(qValues, dim=1)
            # .item converts a single-valued tensor to a normal python number.
            # It only works on single-valued tensors, e.g., tensor([3]). Since
            # qValues only has one row, indices only has one value, so we
            # can use item(). For multi-valued tensors, there's tolist()
            action = int(indices.item())
        return action

    # -----
    # step
    # -----
    @torch.no_grad()
    def step(self, net):
        """
        Chooses an action, performs the action, and then transitions to the
        next state. We don't want these operations to be tracked by torch
        for gradient calculation purposes because these ops have nothing to
        do with the loss function. Also save the experience.
        """
        action = self.choose_action(net)
        nextState, reward, done, _ = self.env.step(action)
        experience = Experience(self.state, action, reward, nextState, done)
        self.memory.add(experience)
        self.state = nextState
        if done:
            self.reset()
