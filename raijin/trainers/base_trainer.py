from abc import ABC
from abc import abstractmethod
from typing import Any
from typing import Dict
from typing import Tuple

from omegaconf.dictconfig import DictConfig
import torch

from raijin.agents import base_agent as ba
from raijin.memory import base_memory as bm
from raijin.utilities.register import register_object


# ============================================
#                BaseTrainer
# ============================================
class BaseTrainer(ABC):
    __name__ = "BaseTrainer"

    # -----
    # subclass_hook
    # -----
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        register_object(cls)

    # -----
    # constructor
    # -----
    def __init__(
        self,
        agent: "ba.BaseAgent",
        memory: "bm.BaseMemory",
        device: str,
        params: DictConfig,
    ) -> None:
        self.agent = agent
        self.memory = memory
        self.device = device
        self.nEpisodes = params.nEpisodes
        self.episodeLength = params.episodeLength
        self.prePopulateSteps = params.prePopulateSteps
        self.batchSize = params.batchSize
        self.discountRate = params.discountRate
        self.episodeOver = False
        self.episodeReward = 0.0
        self.episode = 0
        self.metrics: Dict[str, Any] = {}
        # Prioritized experience replay flag
        if self.memory.__name__ == "PriorityMemory":
            self.usingPER = True
        else:
            self.usingPER = False

    # -----
    # training_step
    # -----
    @abstractmethod
    def training_step(self, actionChoiceType: str) -> None:
        """
        Performs one iteration of the training loop.
        """
        pass

    # -----
    # train_episode
    # -----
    @abstractmethod
    def train_episode(self) -> None:
        """
        Contains the training loop for one full episode.
        """
        pass

    # -----
    # learn
    # -----
    @abstractmethod
    def learn(self, batch: Tuple) -> torch.Tensor:
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
    # episode_start
    # -----
    def episode_start(self) -> None:
        """
        Called at the start of each episode.
        """
        pass

    # -----
    # episode_end
    # -----
    def episode_end(self) -> None:
        """
        Called at the end of each episode.
        """
        pass

    # -----
    # step_start
    # -----
    def step_start(self) -> None:
        """
        Called at the start of each training step. 
        """
        pass

    # -----
    # step_end
    # -----
    def step_end(self) -> None:
        """
        Called at the end of each training step. 
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
