from typing import List

from cleo import Application

from raijin.commands.test import TestCommand
from raijin.commands.train import TrainCommand
from raijin.utilities.config import ApplicationConfig


# ============================================
#              RaijinApplication
# ============================================
class RaijinApplication(Application):
    # -----
    # constructor
    # -----
    def __init__(self) -> None:
        super().__init__(config=ApplicationConfig())
        for command in self._get_commands():
            self.add(command())

    # -----
    # _get_commands
    # -----
    def _get_commands(self) -> List:
        commandList = [
            TestCommand,
            TrainCommand,
        ]
        return commandList
