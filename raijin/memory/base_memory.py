from abc import ABC
from abc import abstractmethod
from typing import Tuple

from raijin.utilities.register import register_object


# ============================================
#                  BaseMemory
# ============================================
class BaseMemory(ABC):
    # -----
    # subclass_hook
    # -----
    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        register_object(cls)

    # -----
    # add
    # -----
    @abstractmethod
    def add(self) -> None:
        """
        Puts a new experience into the memory buffer.
        """
        pass

    # -----
    # sample
    # -----
    @abstractmethod
    def sample(self) -> Tuple:
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
