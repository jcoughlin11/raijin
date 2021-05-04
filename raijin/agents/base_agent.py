from abc import ABC
from abc import abstractmethod

import torch

from raijin.memory.experience import Experience
from raijin.utilities.register import register_object


# ============================================
#                  BaseAgent
# ============================================
class BaseAgent(ABC):
    """
    The agent is the object that interacts with the environment.

    This means that the agent is responsible for:
        * Choosing actions
        * Transitioning to the next state in the game
        * Properly preparing the game frames returned by the environment
            for use in the neural network
    """

    __name__ = "BaseAgent"

    # -----
    # subclass hook
    # -----
    def __init_subclass__(cls) -> None:
        super().__init_subclass__()
        register_object(cls)

    # -----
    # reset
    # -----
    @abstractmethod
    def reset(self) -> None:
        """
        Reverts the environment back to its initial state.
        """
        pass

    # -----
    # choose_action
    # -----
    @abstractmethod
    def choose_action(self, actionChoiceType: str) -> int:
        """
        Selects an action to take.
        """
        pass

    # -----
    # step
    # -----
    @abstractmethod
    def step(self, actionChoiceType: str, net: torch.nn.Module) -> Experience:
        """
        Transitions to the next game state.
        """
        pass

    # -----
    # state_dict
    # -----
    @abstractmethod
    def state_dict(self) -> dict:
        """
        Returns a dictionary containing any stateful parameters. The
        state dictionary is used when saving a checkpoint.
        """
        pass
