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
        pass

    # -----
    # sample
    # -----
    @abstractmethod
    def sample():
        pass
