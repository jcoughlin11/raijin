from abc import ABC
from abc import abstractmethod

from raijin.utilities.register import register_object


# ============================================
#                  BaseMemory
# ============================================
class BaseMemory(ABC):
    # -----
    # subclass_hook
    # -----
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        register_object(cls)

    # -----
    # add
    # -----
    @abstractmethod
    def add():
        """
        Puts a new experience into the memory buffer.
        """
        pass

    # -----
    # sample
    # -----
    @abstractmethod
    def sample():
        """
        Extracts a subset of experiences from the buffer.
        """
        pass

    # -----
    # state_dict
    # -----
    @abstractmethod
    def state_dict():
        pass
