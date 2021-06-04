import os
from typing import List
from typing import Union

from raijin.proctors import base_proctor as bp
from raijin.trainers import base_trainer as bt


# ============================================
#                 MetricList
# ============================================
class MetricList:
    """
    Interface to act on multiple metrics at once.
    """

    __name__ = "MetricList"

    # -----
    # constructor
    # -----
    def __init__(self, metrics: List):
        self.metrics = {m.__name__ : m for m in metrics} 

    # -----
    # reset
    # -----
    def reset(self) -> None:
        for m in self.metrics.values():
            m.reset()

    # -----
    # update
    # -----
    def update(self, mgr: Union["bp.BaseProctor", "bt.BaseTrainer"], when: str) -> None:
        for m in self.metrics.values():
            # This update method is called during each of the
            # trainer/proctor's hooks since various metrics need to be
            # updated at different times. Here we make sure we're only
            # updating the metric if it's the appropriate time
            if m.when == when or m.when == "all":
                m.update(mgr)

    # -----
    # save
    # -----
    def save(self, outputDir) -> None:
        if os.path.basename(outputDir) != "metrics":
            outputDir = os.path.join(outputDir, "metrics")
            os.mkdir(outputDir)
        for m in self.metrics.values():
            m.save(outputDir)

    # -----
    # get
    # -----
    def get(self, metric: str):
        return self.metrics[metric].values
