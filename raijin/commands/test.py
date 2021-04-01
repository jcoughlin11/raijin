import os
from typing import Tuple

from cleo import Command
from clikit.ui.components.progress_bar import ProgressBar
from omegaconf.dictconfig import DictConfig
import torch

from raijin.io.read import read_parameter_file
from raijin.proctors.base_proctor import BaseProctor
from raijin.utilities.managers import get_proctor


# ============================================
#                 TestCommand
# ============================================
class TestCommand(Command):
    """
    Tests the specified agent.

    test
        {path : Path to the directory holding the model and params.}
    """
    # -----
    # handle
    # -----
    def handle(self) -> None:
        """
        See: https://tinyurl.com/6m86cz9s
        and: https://tinyurl.com/358ju8yx
        """
        self.line("<warning>Initializing...</warning>")
        params, proctor, progBar = self._initialize()
        self.line("<warning>Testing...</warning>")
        self.line("\n")
        params, proctor, progBar = self._test(params, proctor, progBar)
        self.line("\n")
        self.line("<warning>Cleaning up...</warning>")
        self._cleanup(params, proctor, progBar)
        self.line("<warning>Done.</warning>")

    # -----
    # _initialize
    # -----
    def _initialize(self) -> Tuple:
        path = self.argument("path")
        params = read_parameter_file(os.path.join(path, "params.yaml"))
        # Load the trained network parameters from the final model file
        modelStateDict = torch.load(os.path.join(path, "model.pt"))
        proctor = get_proctor(params, modelStateDict)
        progBar = self._get_progress_bar(proctor.nEpisodes)
        msg = f"<info>Episode Reward</info>: {proctor.episodeReward}"
        progBar.set_message(msg)
        proctor.pre_test()
        return (params, proctor, progBar)

    # -----
    # _test
    # -----
    def _test(
        self, params: DictConfig, proctor: BaseProctor, progBar: ProgressBar
    ) -> Tuple:
        progBar.start()
        for proctor.episode in range(proctor.nEpisodes):
            proctor.test_step_start()
            proctor.test()
            msg = f"<info>Episode Reward</info>: {proctor.episodeReward}"
            progBar.set_message(msg)
            proctor.test_step_end()
            progBar.advance()
        progBar.finish()
        return (params, proctor, progBar)

    # -----
    # _cleanup
    # -----
    def _cleanup(
        self, params: DictConfig, proctor: BaseProctor, progBar: ProgressBar
    ) -> None:
        proctor.post_test()

    # -----
    # _get_progress_bar
    # -----
    def _get_progress_bar(self, nEpisodes: int) -> ProgressBar:
        progBar = self.progress_bar(nEpisodes)
        formatStr = "\t<info>Episode</info>: %current%/%max%"
        formatStr += "\n\t<info>Elapsed Time</info>: %elapsed%"
        formatStr += "\n\t%message%"
        progBar.set_format(formatStr)
        return progBar
