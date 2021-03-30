from cleo import Application

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
    def _get_commands(self) -> None:
        commandList = [
            TrainCommand,
        ]
        return commandList
