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
    def training_step(self) -> None:
        """
        Performs one iteration of the training loop.
        """
        pass

    # -----
    # train
    # -----
    @abstractmethod
    def train(self) -> None:
        """
        Contains the training loop for one full episode.
        """
        pass

    # -----
    # learn
    # -----
    @abstractmethod
    def learn(self) -> None:
        """
        Updates the weights in the network(s).
        """
        pass

    # -----
    # state_dict
    # -----
    @abstractmethod
    def state_dict(self) -> dict:
        pass

    # -----
    # pre_train
    # -----
    def pre_train(self) -> None:
        """
        Called before the start of training.
        """
        pass

    # -----
    # train_step_start
    # -----
    def train_step_start(self) -> None:
        """
        Called at the start of each episode.
        """
        pass

    # -----
    # train_step_end
    # -----
    def train_step_end(self) -> None:
        """
        Called at the end of each episode.
        """
        pass

    # -----
    # post_train
    # -----
    def post_train(self) -> None:
        """
        Called after training.
        """
        pass
