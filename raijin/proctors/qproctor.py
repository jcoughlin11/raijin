from typing import Any
from typing import Dict
from typing import List

from omegaconf.dictconfig import DictConfig

from raijin.agents import base_agent as ba
from raijin.metrics import metric_list as ml

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
        device: str,
        metrics: "ml.MetricList"
    ) -> None:
        super().__init__(agent, params, device, metrics)
        self.net = nets[0]
        self.net.load_state_dict(modelStateDict)
        self.net.to(self.device)
        # Put the network into evaluation mode
        self.net.eval()

    # -----
    # testing_step
    # -----
    def testing_step(self) -> None:
        self.step_start()
        experience = self.agent.step("exploit", self.net)
        self.episodeReward += experience.reward
        self.episodeOver = experience.done
        self.step_end()

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
    # state_dict
    # -----
    def state_dict(self) -> dict:
        return {}
