from abc import ABC
from abc import abstractmethod
from typing import Union

from raijin.proctors import base_proctor as bp
from raijin.trainers import base_trainer as bt
from raijin.utilities.register import register_object


# ============================================
#                 BaseMetric
# ============================================
class BaseMetric(ABC):
    """
    Exposes a common interface for calculating and/or updating metrics,
    such as the episode reward. Inspired by keras.
    """
    __name__ = "BaseMetric"

    # -----
    # subclass_hook
    # -----
    def __init_subclass__(cls) -> None:
        super().__init_subclass__()
        register_object(cls)

    # -----
    # reset
    # -----
    @abstractmethod
    def reset(self) -> None:
        pass

    # -----
    # update
    # -----
    @abstractmethod
    def update(self, manager: Union["bp.BaseProctor", "bt.BaseTrainer"]) -> None:
        pass

    # -----
    # log
    # -----
    @abstractmethod
    def log(self) -> None:
        pass
