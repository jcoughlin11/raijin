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
        self.metrics = metrics

    # -----
    # reset
    # -----
    def reset(self) -> None:
        for m in self.metrics:
            m.reset()

    # -----
    # update
    # -----
    def update(self, manager: Union["bp.BaseProctor", "bt.BaseTrainer"]) -> None:
        for m in self.metrics:
            m.update(manager)

    # -----
    # save
    # -----
    def save(self, outputDir) -> None:
        if os.path.basename(outputDir) != "metrics":
            outputDir = os.path.join(outputDir, "metrics")
            os.mkdir(outputDir)
        for m in self.metrics:
            m.save(outputDir)
