from abc import ABC
from abc import abstractmethod

from raijin.utilities.register import register_object


# ============================================
#                 BaseDataset
# ============================================
class BaseDataset(ABC):
    # -----
    # subclass hook
    # -----
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        register_object(cls)

    # -----
    # iter
    # -----
    @abstractmethod
    def __iter__(self):
        pass
