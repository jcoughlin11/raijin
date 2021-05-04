import os

from cleo import Command
import gym
import numpy as np
import torch

from raijin.io.read import read_parameter_file
from raijin.utilities.managers import get_nets
from raijin.utilities.register import registry


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
        self._initialize()
        self.line("<warning>Testing...</warning>")
        self.line("\n")
        self._test()
        self.line("\n")
        self.line("<warning>Metrics</warning>:")
        self._cleanup()
        self.line("<warning>Done.</warning>")

    # -----
    # _initialize
    # -----
    def _initialize(self) -> None:
        path = self.argument("path")
        self.params = read_parameter_file(os.path.join(path, "params.yaml"))
        # Load the trained network parameters from the final model file
        modelStateDict = torch.load(os.path.join(path, "model.pt"))
        self._get_proctor(modelStateDict)
        self._get_progress_bar(self.proctor.nEpisodes)

    # -----
    # _test
    # -----
    def _test(self) -> None:
        self.progBar.start()
        self.proctor.pre_test()
        # NOTE: If using a deterministic gym env, the result will be
        # the same for every episode
        if "Deterministic" in self.params.env.name:
            self.proctor.nEpisodes = 1
        for self.proctor.episode in range(self.proctor.nEpisodes):
            self.proctor.test_step_start()
            self.proctor.test_step()
            msg = f"<info>Episode Reward</info>: {self.proctor.episodeReward}"
            self.progBar.set_message(msg)
            self.proctor.test_step_end()
            self.progBar.advance()
        self.progBar.finish()
        self.proctor.post_test()

    # -----
    # _cleanup
    # -----
    def _cleanup(self) -> None:
        maxReward = np.max(self.proctor.metrics["episodeRewards"])
        avgReward = np.mean(self.proctor.metrics["episodeRewards"])
        stdDev = np.std(self.proctor.metrics["episodeRewards"])
        self.line(f"\t<info>Max score</info>: {maxReward}")
        self.line(f"\t<info>Avg. score</info>: {avgReward}")
        self.line(f"\t<info>Std. Dev</info>: {stdDev}")

    # -----
    # _get_proctor
    # -----
    def _get_proctor(self, modelStateDict: dict) -> None:
        env = gym.make(self.params.env.name)
        pipeline = registry[self.params.pipeline.name](self.params.pipeline)
        agent = registry[self.params.agent.name](
            env, pipeline, self.params.agent
        )
        nets = get_nets(self.params.nets, pipeline.traceLen, env.action_space.n)
        self.proctor = registry[self.params.proctor.name](
            agent, nets, modelStateDict, self.params.proctor
        )

    # -----
    # _get_progress_bar
    # -----
    def _get_progress_bar(self, nEpisodes: int) -> None:
        self.progBar = self.progress_bar(nEpisodes)
        formatStr = "\t<info>Episode</info>: %current%/%max%"
        formatStr += "\n\t<info>Elapsed Time</info>: %elapsed%"
        formatStr += "\n\t%message%"
        self.progBar.set_format(formatStr)
        msg = f"<info>Episode Reward</info>: {self.proctor.episodeReward}"
        self.progBar.set_message(msg)
