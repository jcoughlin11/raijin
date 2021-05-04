from abc import ABC
from abc import abstractmethod
from typing import Tuple

from raijin.utilities.register import register_object

from .experience import Experience


# ============================================
#                  BaseMemory
# ============================================
class BaseMemory(ABC):
    __name__ = "BaseMemory"

    # -----
    # subclass_hook
    # -----
    def __init_subclass__(cls) -> None:
        super().__init_subclass__()
        register_object(cls)

    # -----
    # add
    # -----
    @abstractmethod
    def add(self, experience: Experience) -> None:
        """
        Puts a new experience into the memory buffer.
        """
        pass

    # -----
    # sample
    # -----
    @abstractmethod
    def sample(self, batchSize: int) -> Tuple:
        """
        Extracts a subset of experiences from the buffer.
        """
        pass

    # -----
    # state_dict
    # -----
    @abstractmethod
    def state_dict(self) -> dict:
        pass

    # -----
    # update
    # -----
    def update(self) -> None:
        pass
