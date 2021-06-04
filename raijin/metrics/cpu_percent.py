import os

import psutil
import yaml

from raijin.proctors import base_proctor as bp
from raijin.trainers import base_trainer as bt

from .base_metric import BaseMetric


# ============================================
#                  CPUPercent
# ============================================
class CPUPercent(BaseMetric):

    __name__ = "CPUPercent"

    # -----
    # constructor
    # -----
    def __init__(self):
        self.cpuPercent = {} 

    # -----
    # reset
    # -----
    def reset(self) -> None:
        self.cpuPercent = {} 

    # -----
    # update
    # -----
    def update(self, mgr: Union["bp.BaseProctor", "bt.BaseTrainer"]) -> None:
        self.cpuPercent[f"mgr.episode"] = psutil.cpu_percent(0.25)

    # -----
    # save
    # -----
    def save(self, outputDir: str) -> None:
        with open(os.path.join(outputDir, "cpu_percent.yaml"), "w") as fd:
            yaml.safe_dump(self.cpuPercent, fd)
