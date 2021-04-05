from gym import Env
from omegaconf.dictconfig import DictConfig
import torch

from raijin.memory.experience import Experience
from raijin.pipelines import base_pipeline as bp

from .base_agent import BaseAgent


# ============================================
#                 RandomAgent
# ============================================
class RandomAgent(BaseAgent):
    """
    An agent that performs only randomly selected actions. 

    * Progresses through the game frame-by-frame
    * Employs an epsilon-greedy strategy for action selection
    """

    __name__ = "RandomAgent"

    # -----
    # constructor
    # -----
    def __init__(
        self, env: Env, pipeline: "bp.BasePipeline", params: DictConfig
    ) -> None:
        self.env = env
        self.pipeline = pipeline
        self.state = None

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
    def choose_action(self, actionChoiceType: str) -> int:
        """
        Selects a random action.
        """
        return self.env.action_space.sample()

    # -----
    # step
    # -----
    @torch.no_grad()
    def step(self, actionChoiceType: str, net: torch.nn.Module) -> Experience:
        """
        Transition from one game frame to the next.
        """
        action = self.choose_action("explore")
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
        return {}
