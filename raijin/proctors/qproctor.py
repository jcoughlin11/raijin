from typing import Any
from typing import Dict
from typing import List

from omegaconf.dictconfig import DictConfig

from raijin.agents import base_agent as ba

from .base_proctor import BaseProctor


# ============================================
#                   QProctor
# ============================================
class QProctor(BaseProctor):
    __name__ = "QProctor"

    # -----
    # constructor
    # -----
    def __init__(
        self,
        agent: "ba.BaseAgent",
        nets: List,
        modelStateDict: dict,
        params: DictConfig,
        metrics
    ) -> None:
        self.agent = agent
        self.net = nets[0]
        self.net.load_state_dict(modelStateDict)
        self.nEpisodes = params.nEpisodes
        self.episodeLength = params.episodeLength
        self.metrics = metrics
        self.episodeOver = False
        self.episodeReward = 0.0
        self.episode = 0
        # Put the network into evaluation mode
        self.net.eval()

    # -----
    # pre_test
    # -----
    def pre_test(self) -> None:
        self.metrics.reset()

    # -----
    # testing_step
    # -----
    def testing_step(self) -> None:
        experience = self.agent.step("exploit", self.net)
        self.episodeReward += experience.reward
        self.episodeOver = experience.done

    # -----
    # test_episode
    # -----
    def test_episode(self) -> None:
        self.agent.reset()
        for episodeStep in range(self.episodeLength):
            self.testing_step()
            if self.episodeOver:
                break

    # -----
    # step_end
    # -----
    def step_end(self) -> None:
        self.episodeOver = False
        self.metrics.update(self)
        self.episodeReward = 0.0

    # -----
    # state_dict
    # -----
    def state_dict(self) -> dict:
        return {}
