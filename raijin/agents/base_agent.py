from abc import ABC
from abc import abstractmethod

from raijin.utilities.register import register_object


# ============================================
#                  BaseAgent
# ============================================
class BaseAgent(ABC):
    # -----
    # subclass hook
    # -----
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        register_object(cls)

    # -----
    # reset
    # -----
    @abstractmethod
    def reset(self):
        pass

    # -----
    # choose_action
    # -----
    @abstractmethod
    def choose_action(self):
        pass

    # -----
    # step
    # -----
    @abstractmethod
    def step(self):
        pass

    # -----
    # state_dict
    # -----
    @abstractmethod
    def state_dict():
        pass
