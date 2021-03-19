from cleo import Command
import numpy as np

from raijin.io.read import read_parameter_file
from raijin.utilities.managers import get_trainer


# ============================================
#                TrainCommand
# ============================================
class TrainCommand(Command):
    """
    Trains an agent according to the given parameter file.

    train
        {paramFile : Yaml file containing run parameters.}
    """
    # -----
    # handle
    # -----
    def handle(self):
        self.line("<warning>Training...</warning>")
        self.line("\n")
        params = read_parameter_file(self.argument("paramFile"))
        trainer = get_trainer(params)
        progressBar = self._get_progress_bar(trainer.nEpisodes)
        progressBar.set_message(f"<info>Episode Reward</info>: {trainer.episodeReward}")
        progressBar.start()
        trainer.pre_train()
        for episode in range(trainer.nEpisodes):
            trainer.train_step_start()
            trainer.train()
            progressBar.set_message(f"<info>Episode Reward</info>: {trainer.episodeReward}")
            trainer.train_step_end()
            progressBar.advance()
        trainer.post_train()
        progressBar.finish()

    # -----
    # _get_progress_bar
    # -----
    def _get_progress_bar(self, nEpisodes):
        progressBar = self.progress_bar(nEpisodes)
        formatStr = "\t<info>Episode</info>: %current%/%max%"
        formatStr += "\n\t<info>Elapsed Time</info>: %elapsed%"
        formatStr += "\n\t%message%"
        progressBar.set_format(formatStr)
        return progressBar
