from abc import ABC
from abc import abstractmethod

from raijin.utilities.register import register_object


# ============================================
#               BasePipeline
# ============================================
class BasePipeline(ABC):
    # -----
    # subclass_hook
    # -----
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        register_object(cls)

    # -----
    # processs
    # -----
    @abstractmethod
    def process():
        pass
