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
    def update(self, reward=0.0) -> None:
        self.episodeRewards.append(reward)

    # -----
    # log
    # -----
    def log(self) -> None:
        pass
