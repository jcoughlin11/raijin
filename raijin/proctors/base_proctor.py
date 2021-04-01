from abc import ABC
from abc import abstractmethod

from raijin.utilities.register import register_object


# ============================================
#                BaseProctor
# ============================================
class BaseProctor(ABC):
    # -----
    # subclass_hook
    # -----
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        register_object(cls)

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
    # test
    # -----
    @abstractmethod
    def test(self) -> None:
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
        pass

    # -----
    # test_step_start
    # -----
    def test_step_start(self) -> None:
        """
        Called at the start of each episode.
        """
        pass

    # -----
    # test_step_end
    # -----
    def test_step_end(self) -> None:
        """
        Called at the end of each episode.
        """
        pass

    # -----
    # post_test
    # -----
    def post_test(self) -> None:
        """
        Called after testing.
        """
        pass
