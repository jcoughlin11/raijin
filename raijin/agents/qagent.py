import numpy as np
import torch

from raijin.memory.experience import Experience

from .base_agent import BaseAgent


# ============================================
#                    QAgent
# ============================================
class QAgent(BaseAgent):
    __name__ = "QAgent"
    # -----
    # constructor
    # -----
    def __init__(self, env, pipeline, params):
        self.env = env
        self.pipeline = pipeline
        self.epsilonStart = params.epsilonStart
        self.epsilonStop = params.epsilonStop
        self.epsilonDecayRate = params.epsilonDecayRate
        self.state = None
        self.decayStep = 0

    # -----
    # reset
    # -----
    def reset(self):
        # env produces an array of shape (H, W, C), so need to reshape
        frame = self.env.reset()
        frame = self._reshape_frame(frame) 
        self.state = self.pipeline.process(frame, True)

    # -----
    # choose_action
    # -----
    def choose_action(self, actionChoiceType, net):
        if actionChoiceType == "train":
            exploitProb = np.random.random()
            exploreProb = self.epsilonStop + (
                self.epsilonStart - self.epsilonStop
            ) * np.exp(-self.epsilonDecayRate * self.decayStep)
            self.decayStep += 1
            if exploreProb >= exploitProb:
                actionChoiceType = "explore"
            else:
                actionChoiceType = "exploit"
        if actionChoiceType == "explore":
            action = self.env.action_space.sample()
        elif actionChoiceType == "exploit":
            # state has shape (C, H, W), but needs shape (N, C, H, W) even
            # with only one sample
            state = torch.unsqueeze(self.state, 0)
            action = torch.argmax(net(state)).item()
        return action

    # -----
    # step
    # -----
    def step(self, actionChoiceType, net):
        action = self.choose_action(actionChoiceType, net)
        nextFrame, reward, done, _ = self.env.step(action)
        # env produces an array of shape (H, W, C), so need to reshape
        nextFrame = self._reshape_frame(nextFrame)
        nextState = self.pipeline.process(nextFrame, False)
        experience = Experience(self.state, action, reward, nextState, done)
        if done:
            self.reset()
        else:
            self.state = nextState
        return experience

    # -----
    # _reshape_frame
    # -----
    def _reshape_frame(self, frame):
        return frame.reshape([frame.shape[-1],] + list(frame.shape[:-1])) 

    # -----
    # state_dict
    # -----
    def state_dict(self):
        stateDict = {
            "envState" : self.env.clone_full_state(), 
            "pipeline" : self.pipeline.state_dict(),
            "state" : self.state,
            "decayStep" : self.decayStep
        }
        return stateDict
