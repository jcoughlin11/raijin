from gym import Env
from omegaconf.dictconfig import DictConfig
import torch

from raijin.memory.experience import Experience
from raijin.pipelines import base_pipeline as bp

from .qagent import QAgent


# ============================================
#                 RandomAgent
# ============================================
class RandomAgent(QAgent):
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
        self,
        env: Env,
        pipeline: "bp.BasePipeline",
        params: DictConfig,
        device: str,
    ) -> None:
        super().__init__(env, pipeline, params, device)

    # -----
    # state_dict
    # -----
    def state_dict(self) -> dict:
        return {}
