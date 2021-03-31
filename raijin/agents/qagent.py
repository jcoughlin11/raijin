import numpy as np
from gym import Env
from omegaconf.dictconfig import DictConfig
import torch

from raijin.memory.experience import Experience
from raijin.pipelines import base_pipeline as bp

from .base_agent import BaseAgent


# ============================================
#                    QAgent
# ============================================
class QAgent(BaseAgent):
    """
    The agent described in [Mnih et al. 2013][1].

    * Progresses through the game frame-by-frame
    * Employs an epsilon-greedy strategy for action selection

    [1]: https://arxiv.org/abs/1312.5602
    """

    __name__ = "QAgent"

    # -----
    # constructor
    # -----
    def __init__(
        self, env: Env, pipeline: "bp.BasePipeline", params: DictConfig
    ) -> None:
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
    def reset(self) -> None:
        """
        Reverts the environment back to its initial state.
        """
        frame = self.env.reset()
        self.state = self.pipeline.process(frame, True)

    # -----
    # choose_action
    # -----
    def choose_action(self, actionChoiceType: str, net: torch.nn.Module) -> int:
        """
        Implements epsilon-greedy action selection strategy.

        * Draw a random number `n` from a uniform distribution in [0,1)
        * Calculate epsilon for the current timestep
        * If n < epsilon, we choose a random action
        * Otherwise we go with the action the network thinks is best

        Epsilon starts large so that, at the beginning of training, we
        have a high probability of picking a random action. This allows
        the agent to explore a large number of state-action pairs.

        As time goes on and the agent has learned more, epsilon gets
        smaller so that there is a higher probability of selecting a
        "known good" action and, therefore, progressing further into
        the game.
        """
        if actionChoiceType == "train":
            n = np.random.random()
            epsilon = self.epsilonStop + (
                self.epsilonStart - self.epsilonStop
            ) * np.exp(-self.epsilonDecayRate * self.decayStep)
            self.decayStep += 1
            if n <= epsilon:
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
    def step(self, actionChoiceType: str, net: torch.nn.Module) -> Experience:
        """
        Transition from one game frame to the next.
        """
        action = self.choose_action(actionChoiceType, net)
        nextFrame, reward, done, _ = self.env.step(action)
        nextState = self.pipeline.process(nextFrame, False)
        experience = Experience(self.state, action, reward, nextState, done)
        if done:
            self.reset()
        else:
            self.state = nextState
        return experience

    # -----
    # state_dict
    # -----
    def state_dict(self) -> dict:
        stateDict = {
            "envState": self.env.clone_full_state(),
            "pipeline": self.pipeline.state_dict(),
            "state": self.state,
            "decayStep": self.decayStep,
        }
        return stateDict
