import os

import psutil
import yaml

from raijin.proctors import base_proctor as bp
from raijin.trainers import base_trainer as bt

from .base_metric import BaseMetric


# ============================================
#                   RAMUsage 
# ============================================
class RAMUsage(BaseMetric):

    __name__ = "RAMUsage"

    # -----
    # constructor
    # -----
    def __init__(self):
        self.ramUsage = {} 

    # -----
    # reset
    # -----
    def reset(self) -> None:
        self.ramUsage = {} 

    # -----
    # update
    # -----
    def update(self, mgr: Union["bp.BaseProctor", "bt.BaseTrainer"]) -> None:
        self.ramUsage[f"mgr.episode"] = psutil.virtual_memory()[2]

    # -----
    # save
    # -----
    def save(self, outputDir: str) -> None:
        with open(os.path.join(outputDir, "ram_usage.yaml"), "w") as fd:
            yaml.safe_dump(self.ramUsage, fd)
