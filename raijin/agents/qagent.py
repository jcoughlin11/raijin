import numpy as np

from .base_agent import BaseAgent


# ============================================
#                    QAgent
# ============================================
class QAgent(BaseAgent):
    # -----
    # constructor
    # -----
    def __init__(self, env, params):
        self.env = env
        self.epsilonStart = params.epsilonStart
        self.epsilonStop = params.epsilonStop
        self.epsilonDecayRate = params.epsilonDecayRate
        self.state = None
        self.decayStep = 0

    # -----
    # reset
    # -----
    def reset(self):
        self.state = self.env.reset()

    # -----
    # choose_action
    # -----
    def choose_action(self, actionChoiceType, net):
        if actionChoiceType == "train":
            exploitProb = np.random.random()
            exploreProb = self.epsilonStop + (
                self.epsilonStart - self.epsilonStop
            ) * np.exp(-self.epsDecayRate * self.decayStep)
            self.decayStep += 1
            if exploreProb >= exploitProb:
                actionChoiceType = "explore"
            else:
                actionChoiceType = "exploit"
        if actionChoiceType == "explore":
            action = self.env.action_space.sample()
        elif actionChoiceType == "exploit":
            action = net.predict(self.state)
        return action

    # -----
    # step
    # -----
    def step(self, actionChoiceType, net):
        action = self.choose_action(actionChoiceType, net)
        nextFrame, reward, done, _ = self.env.step(action)
        nextState = self.pipeline.process(nextFrame, False)
        experience = Experience(self.state, action, reward, nextState, done)
        self.state = nextState
        return experience
