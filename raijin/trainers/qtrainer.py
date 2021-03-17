import torch

from raijin.utilities.register import registry

from .base_trainer import BaseTrainer


# ============================================
#                   QTrainer
# ============================================
class QTrainer(BaseTrainer):
    # -----
    # constructor
    # -----
    def __init__(self, agent, lossFunctions, memory, nets, optimizers, params, pipeline):
        self.agent = agent
        self.loss_function = lossFunctions[0]  
        self.memory = memory
        self.nets = nets[0]
        self.optimzers = optimizers[0]
        self.pipeline = pipeline
        self.nEpisodes = params.nEpisodes
        self.episodeLength = params.episodeLength
        self.prePopulateSteps = params.prePopulateSteps
        self.batchSize = params.batchSize

    # -----
    # training_step
    # -----
    def training_step(self, actionChoiceType):
        experience = self.agent.step(actionChoiceType, self.net)
        self.memory.add(experience)

    # -----
    # train
    # -----
    def train(self):
        self._pre_populate()
        for episode in range(self.nEpisodes):
            self.agent.reset()
            for episodeStep in range(self.episodeLength):
                self.training_step("train")
                batch = self.memory.sample(self.batchSize)
                self.learn(batch)

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
        targets = self._get_targets()
        loss = self.loss_function(beliefs, targets.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    # -----
    # _pre_populate
    # -----
    def _pre_populate(self):
        self.agent.reset()
        for _ in range(self.prePopulateSteps)
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
        # oneHot.shape should be (N, nActions)
        oneHot = qVals * torch.nn.functional.one_hot(actions, nActions).squeeze()
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
        targets = rewards + self.params.discountRate * maskedVals
        return targets

