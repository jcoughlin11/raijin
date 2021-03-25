import torch

from raijin.utilities.register import registry

from .base_trainer import BaseTrainer


# ============================================
#                   QTrainer
# ============================================
class QTrainer(BaseTrainer):
    __name__ = "QTrainer"
    # -----
    # constructor
    # -----
    def __init__(self, agent, lossFunctions, memory, nets, optimizers, params):
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
    def pre_train(self):
        self._pre_populate()

    # -----
    # training_step
    # -----
    def training_step(self, actionChoiceType):
        experience = self.agent.step(actionChoiceType, self.net)
        self.episodeReward += experience.reward
        self.memory.add(experience)
        self.episodeOver = experience.done

    # -----
    # train
    # -----
    def train(self):
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
    def train_step_end(self):
        self.episodeOver = False
        self.episodeRewards.append(self.episodeReward)
        self.episodeReward = 0.0

    # -----
    # learn
    # -----
    def learn(self, batch):
        """
        states.shape should be (N, traceLen, rows, cols)
        actions.shape should be (N, 1)
        rewards.shape should be (N, 1)
        nextStates.shape should == states.shape
        dones.shape should be (N, 1)
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
    def _pre_populate(self):
        self.agent.reset()
        for _ in range(self.prePopulateSteps):
            self.training_step("explore")

    # -----
    # _get_beliefs
    # -----
    def _get_beliefs(self, states, actions):
        # qVals should have shape (N, nActions)
        qVals = self.net(states)
        nActions = qVals.shape[1]
        # We only need to change the Q-values for the chosen actions,
        # so we use a one-hot to vectorize the calculation while
        # simultaneously keeping the non-chosen indices untouched
        # one_hot returns a tensor with one more dimension than the input,
        # se we squeeze that out
        # one_hot has to have a long dtype. torch.int is int32, which gives
        # an error
        # oneHot.shape should be (N, nActions)
        oneHot = qVals * torch.nn.functional.one_hot(actions.to(torch.int64), nActions).squeeze()
        # qChosen should have shape (N, 1)
        qChosen = torch.sum(oneHot, 1, keepdims=True)
        return qChosen

    # -----
    # _get_targets
    # -----
    def _get_targets(self, nextStates, dones, rewards):
        # Uses Bellman equation to get "right" answers. Shape should be
        # (N, nActions)
        qNext = self.net(nextStates)
        # the max operation doesn't return a tensor, it returns an object
        # that contains both the values and indices. We want the values,
        # which is a tensor. qNextMax should be (N, 1)
        qNextMax = torch.max(qNext, 1, keepdims=True).values
        maskedVals = (1.0 - dones) * qNextMax
        targets = rewards + self.discountRate * maskedVals
        return targets

    # -----
    # state_dict
    # -----
    def state_dict(self):
        return {}
