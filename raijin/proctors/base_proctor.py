from abc import ABC
from abc import abstractmethod

from omegaconf.dictconfig import DictConfig

from raijin.agents import base_agent as ba
from raijin.metrics import metric_list as ml
from raijin.utilities.register import register_object


# ============================================
#                BaseProctor
# ============================================
class BaseProctor(ABC):
    __name__ = "BaseProctor"

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
        params: DictConfig,
        device: str,
        metrics: "ml.MetricList"
    ) -> None:
        self.agent = agent
        self.nEpisodes = params.nEpisodes
        self.episodeLength = params.episodeLength
        self.device = device
        self.metrics = metrics
        self.episodeOver = False
        self.episodeReward = 0.0
        self.episode = 0

    # -----
    # testing_step
    # -----
    @abstractmethod
    def testing_step(self) -> None:
        """
        Performs one iteration of the test loop.
        """
        pass

    # -----
    # test_episode
    # -----
    @abstractmethod
    def test_episode(self) -> None:
        """
        Contains the testing loop for one full episode.
        """
        pass

    # -----
    # state_dict
    # -----
    @abstractmethod
    def state_dict(self) -> dict:
        pass

    # -----
    # pre_test
    # -----
    def pre_test(self) -> None:
        """
        Called before the start of testing.
        """
        self.metrics.reset()

    # -----
    # episode_start
    # -----
    def episode_start(self) -> None:
        """
        Called at the start of each episode.
        """
        self.metrics.update(self, "episode_start")

    # -----
    # episode_end
    # -----
    def episode_end(self) -> None:
        """
        Called at the end of each episode.
        """
        self.episodeOver = False
        self.metrics.update(self, "episode_end")
        self.episodeReward = 0.0

    # -----
    # step_start
    # -----
    def step_start(self) -> None:
        """
        Called at the start of each training step. 
        """
        self.metrics.update(self, "step_start")

    # -----
    # step_end
    # -----
    def step_end(self) -> None:
        """
        Called at the end of each training step. 
        """
        self.metrics.update(self, "step_end")

    # -----
    # post_test
    # -----
    def post_test(self) -> None:
        """
        Called after testing.
        """
        pass
