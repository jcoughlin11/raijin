import os

from cleo import Command
import gym
import numpy as np
import torch

from raijin.io.read import read_parameter_file
from raijin.io.write import display_banner
from raijin.metrics.metric_list import MetricList
from raijin.utilities.io_utilities import get_numbered_dir
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
        display_banner(self.line)
        self.line("<warning>Initializing...</warning>")
        self._setup()
        self.line("<warning>Testing...</warning>")
        self.line("\n")
        self._test()
        self.line("\n")
        self.line("<warning>Metrics</warning>:")
        self._cleanup()
        self.line("<warning>Done.</warning>")

    # -----
    # _setup
    # -----
    def _setup(self) -> None:
        self.path = self.argument("path")
        self.params = read_parameter_file(os.path.join(self.path, "params.yaml"))
        # Load the trained network parameters from the final model file
        modelStateDict = torch.load(os.path.join(self.path, "model.pt"))
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
            self.proctor.episode_start()
            self.proctor.test_episode()
            msg = f"<info>Episode Reward</info>: {self.proctor.episodeReward}"
            self.progBar.set_message(msg)
            self.proctor.episode_end()
            self.progBar.advance()
        self.proctor.post_test()
        self.progBar.finish()

    # -----
    # _cleanup
    # -----
    def _cleanup(self) -> None:
        # Print metrics to stdout
        episodeRewards = self.proctor.metrics.get("EpisodeReward")
        self.line(f"\t<info>Max score</info>: {np.max(episodeRewards)}")
        self.line(f"\t<info>Avg. score</info>: {np.mean(episodeRewards)}")
        self.line(f"\t<info>Std. Dev</info>: {np.std(episodeRewards)}")
        # Save metrics
        self.proctor.metrics.save(get_numbered_dir(self.path, "evaluation"))

    # -----
    # _get_proctor
    # -----
    def _get_proctor(self, modelStateDict: dict) -> None:
        device = self.params.device.name
        env = gym.make(self.params.env.name)
        pipeline = registry[self.params.pipeline.name](self.params.pipeline)
        agent = registry[self.params.agent.name](
            env, pipeline, self.params.agent, device
        )
        nets = get_nets(self.params.nets, pipeline.traceLen, env.action_space.n)
        metrics = MetricList([registry[m]() for m in self.params.metrics])
        self.proctor = registry[self.params.proctor.name](
            agent, nets, modelStateDict, self.params.proctor, device, metrics
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
