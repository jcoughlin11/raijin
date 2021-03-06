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
    def __init__(self, agent: "ba.BaseAgent", nets: List, modelStateDict: dict, params: DictConfig) -> None:
        self.agent = agent
        self.net = nets[0]
        self.net.load_state_dict(modelStateDict)
        self.nEpisodes = params.nEpisodes
        self.episodeLength = params.episodeLength
        self.episodeOver = False
        self.episodeReward = 0.0
        self.episode = 0
        self.metrics = {}
        # Put the network into evaluation mode
        self.net.eval()

    # -----
    # pre_test
    # -----
    def pre_test(self) -> None:
        self._initialize_metrics()

    # -----
    # testing_step
    # -----
    def testing_step(self) -> None:
        experience = self.agent.step("exploit", self.net)
        self.episodeReward += experience.reward
        self.episodeOver = experience.done

    # -----
    # test
    # -----
    def test(self) -> None:
        self.agent.reset()
        for episodeStep in range(self.episodeLength):
            self.testing_step()
            if self.episodeOver:
                break

    # -----
    # test_step_end
    # -----
    def test_step_end(self) -> None:
        self.episodeOver = False
        self.metrics["episodeRewards"].append(self.episodeReward)
        self.episodeReward = 0.0

    # -----
    # _initialize_metrics
    # -----
    def _initialize_metrics(self) -> None:
        self.metrics["episodeRewards"] = []

    # -----
    # state_dict
    # -----
    def state_dict(self) -> dict:
        return {}
