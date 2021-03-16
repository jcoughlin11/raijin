from abc import ABC
from abc import abstractmethod

from raijin.utilities.register import register_object


# ============================================
#                BaseTrainer
# ============================================
class BaseTrainer(ABC):
    # -----
    # subclass_hook
    # -----
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        register_object(cls)

    # -----
    # training_step
    # -----
    @abstractmethod
    def training_step():
        pass

    # -----
    # train
    # -----
    @abstractmethod
    def train():
        pass

    # -----
    # learn
    # -----
    @abstractmethod
    def learn():
        pass
