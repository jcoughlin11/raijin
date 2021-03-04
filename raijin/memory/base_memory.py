from abc import ABC
from abc import abstractmethod

from raijin.utilities.register import register_object


# ============================================
#                 BaseMemory
# ============================================
class BaseMemory(ABC):
    # -----
    # subclass hook
    # -----
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        register_object(cls)

    # -----
    # size
    # -----
    @abstractmethod
    def __len__(self):
        pass

    # -----
    # add
    # -----
    @abstractmethod
    def add(self):
        pass

    # -----
    # sample
    # -----
    @abstractmethod
    def sample(self):
        pass
