from cleo import Command

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
        for episode in range(trainer.nEpisodes):
            trainer.train()
            progressBar.advance()
        progressBar.finish()

    # -----
    # _get_progress_bar
    # -----
    def _get_progress_bar(self, nEpisodes):
        progressBar = self.progress_bar(nEpisodes)
        formatStr = "\t<info>Episode</info>: %current%/%max%"
        formatStr += "\n\t<info>Elapsed Time</info>: %elapsed%"
        progressBar.set_format(formatStr)
        return progressBar
