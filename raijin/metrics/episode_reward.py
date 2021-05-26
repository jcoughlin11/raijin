from typing import Union

from raijin.proctors import base_proctor as bp
from raijin.trainers import base_trainer as bt

from .base_metric import BaseMetric


# ============================================
#               EpisodeReward
# ============================================
class EpisodeReward(BaseMetric):

    __name__ = "EpisodeReward"

    # -----
    # constructor
    # -----
    def __init__(self):
        self.episodeRewards = []

    # -----
    # reset
    # -----
    def reset(self) -> None:
        self.episodeRewards = []

    # -----
    # update
    # -----
    def update(self, manager: Union["bp.BaseProctor", "bt.BaseTrainer"]) -> None:
        self.episodeRewards.append(manager.episodeReward)

    # -----
    # log
    # -----
    def log(self) -> None:
        pass
