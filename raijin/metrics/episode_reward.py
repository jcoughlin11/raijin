import os
from typing import Union

import yaml

from raijin.proctors import base_proctor as bp
from raijin.trainers import base_trainer as bt

from .base_metric import BaseMetric


# ============================================
#               EpisodeReward
# ============================================
class EpisodeReward(BaseMetric):

    __name__ = "EpisodeReward"
    when = "episode_end"

    # -----
    # constructor
    # -----
    def __init__(self) -> None:
        self.episodeRewards = {} 

    # -----
    # reset
    # -----
    def reset(self) -> None:
        self.episodeRewards = {} 

    # -----
    # update
    # -----
    def update(self, mgr: Union["bp.BaseProctor", "bt.BaseTrainer"]) -> None:
        self.episodeRewards[f"{mgr.episode}"] = mgr.episodeReward

    # -----
    # save
    # -----
    def save(self, outputDir) -> None:
        with open(os.path.join(outputDir, "episode_rewards.yaml"), "w") as fd:
            yaml.safe_dump(self.episodeRewards, fd)

    # -----
    # values
    # -----
    @property
    def values(self) -> dict:
        return list(self.episodeRewards.values())
