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

    # -----
    # state_dict
    # -----
    @abstractmethod
    def state_dict():
        pass

    # -----
    # pre_train
    # -----
    def pre_train(self):
        pass

    # -----
    # train_step_start
    # -----
    def train_step_start(self):
        pass
        
    # -----
    # train_step_end
    # -----
    def train_step_end(self):
        pass
        
    # -----
    # post_train
    # -----
    def post_train(self):
        pass
