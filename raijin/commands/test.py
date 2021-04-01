import os
from typing import Tuple

from cleo import Command
from clikit.ui.components.progress_bar import ProgressBar
import numpy as np
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
        self.line("<warning>Metrics</warning>:")
        self._cleanup(params, proctor)
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
        # NOTE: If using a deterministic gym env, the result will be
        # the same for every episode
        if "Deterministic" in params.env.name:
            proctor.nEpisodes = 1
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
    def _cleanup(self, params: DictConfig, proctor: BaseProctor) -> None:
        maxReward = np.max(proctor.metrics["episodeRewards"])
        avgReward = np.mean(proctor.metrics["episodeRewards"])
        stdDev = np.std(proctor.metrics["episodeRewards"])
        self.line(f"\t<info>Max score</info>: {maxReward}")
        self.line(f"\t<info>Avg. score</info>: {avgReward}")
        self.line(f"\t<info>Std. Dev</info>: {stdDev}")
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
