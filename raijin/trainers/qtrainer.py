from typing import List
from typing import Tuple

from omegaconf.dictconfig import DictConfig
import torch

from raijin.agents.base_agent import BaseAgent
from raijin.memory.base_memory import BaseMemory

from .base_trainer import BaseTrainer


# ============================================
#                   QTrainer
# ============================================
class QTrainer(BaseTrainer):
    __name__ = "QTrainer"

    # -----
    # constructor
    # -----
    def __init__(self, agent: BaseAgent, lossFunctions: List, memory: BaseMemory, nets: List, optimizers: List, params: DictConfig) -> None:
        self.agent = agent
        self.loss_function = lossFunctions[0]
        self.memory = memory
        self.net = nets[0]
        self.optimizer = optimizers[0]
        self.nEpisodes = params.nEpisodes
        self.episodeLength = params.episodeLength
        self.prePopulateSteps = params.prePopulateSteps
        self.batchSize = params.batchSize
        self.discountRate = params.discountRate
        self.episodeOver = False
        self.episodeReward = 0.0
        self.episodeRewards = []

    # -----
    # pre_train
    # -----
    def pre_train(self) -> None:
        self._pre_populate()

    # -----
    # training_step
    # -----
    def training_step(self, actionChoiceType: str) -> None:
        experience = self.agent.step(actionChoiceType, self.net)
        self.episodeReward += experience.reward
        self.memory.add(experience)
        self.episodeOver = experience.done

    # -----
    # train
    # -----
    def train(self) -> None:
        self.agent.reset()
        for episodeStep in range(self.episodeLength):
            self.training_step("train")
            batch = self.memory.sample(self.batchSize)
            self.learn(batch)
            if self.episodeOver:
                break

    # -----
    # train_step_end
    # -----
    def train_step_end(self) -> None:
        self.episodeOver = False
        self.episodeRewards.append(self.episodeReward)
        self.episodeReward = 0.0

    # -----
    # learn
    # -----
    def learn(self, batch: Tuple) -> None:
        """
        Implements the Deep-Q Learning algorithm from
        [Mnih et al. 2013][1].

        We compare what the network thinks are the best actions for the
        given states (the beliefs) to the "actual" best actions (the
        targets).

        The targets are determined from the Bellman equation. This is a
        bootstrapping process that utilizes the network's own knowledge
        and, presumably, gets better with time.

        The cost function J depends on the beliefs and the targets.
        Since both of those are determined from the network, they
        depend on the network's parameters W. Since we are taking
        dJ/dW we need to tell pytorch to "pretend" that the targets do
        not depend on the parameters. This is because the targets
        represent the "right" answers and aren't "supposed" to depend
        on the weights. We do this by using `detach` when getting the
        loss.

        [1]: https://arxiv.org/abs/1312.5602
        """
        states, actions, rewards, nextStates, dones = batch
        beliefs = self._get_beliefs(states, actions)
        targets = self._get_targets(nextStates, dones, rewards)
        loss = self.loss_function(beliefs, targets.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    # -----
    # _pre_populate
    # -----
    def _pre_populate(self) -> None:
        """
        Fills the initially empty memory buffer so that we are not
        trying to sample from an empty or under-filled buffer at
        the start of training.
        """
        self.agent.reset()
        for _ in range(self.prePopulateSteps):
            self.training_step("explore")

    # -----
    # _get_beliefs
    # -----
    def _get_beliefs(self, states: torch.Tensor, actions: torch.Tensor) torch.Tensor:
        """
        Gets what the network believes to be the best actions for each
        given state. The strength of this belief is given by the
        Q-value.

        We only need to change the Q-values for the chosen actions,
        so we use a one-hot vector to vectorize the calculation.
        """
        qVals = self.net(states)
        nActions = qVals.shape[1]
        # one_hot returns a tensor with one more dimension than the input,
        # so we squeeze that out. The input also has to have a long dtype.
        oneHot = (
            qVals
            * torch.nn.functional.one_hot(
                actions.to(torch.int64), nActions
            ).squeeze()
        )
        qChosen = torch.sum(oneHot, 1, keepdims=True)
        return qChosen

    # -----
    # _get_targets
    # -----
    def _get_targets(self, nextStates: torch.Tensor, dones: torch.Tensor, rewards: torch.Tensor) -> torch.Tensor:
        """
        Uses the Bellman equation along with the network in order to
        bootstrap the "actual" best action for a given state.

        Getting the targets is actually piecewise; if the result of
        an action is a terminal state, then the target is just the
        reward. Otherwise, we use the Bellman equation. Using a
        mask allows us to do both parts of the calculation at once.
        """
        qNext = self.net(nextStates)
        # the max operation doesn't return a tensor; it returns an object
        # that contains both the values and indices
        qNextMax = torch.max(qNext, 1, keepdims=True).values
        maskedVals = (1.0 - dones) * qNextMax
        targets = rewards + self.discountRate * maskedVals
        return targets

    # -----
    # state_dict
    # -----
    def state_dict(self) -> dict:
        return {}
