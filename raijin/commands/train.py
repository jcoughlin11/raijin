from typing import Tuple

from cleo import Command
from clikit.ui.components.progress_bar import ProgressBar
from omegaconf.dictconfig import DictConfig

from raijin.io.read import read_parameter_file
from raijin.io.write import save_checkpoint
from raijin.io.write import save_final_model
from raijin.trainers.base_trainer import BaseTrainer
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
    def handle(self) -> None:
        self.line("<warning>Initializing...</warning>")
        params, trainer, progBar = self._initialize()
        self.line("<warning>Training...</warning>")
        self.line("\n")
        params, trainer, progBar = self._train(params, trainer, progBar)
        self.line("\n")
        self.line("<warning>Cleaning up...</warning>")
        self._cleanup(params, trainer, progBar)
        self.line("<warning>Done.</warning>")

    # -----
    # _initialize
    # -----
    def _initialize(self) -> Tuple:
        params = read_parameter_file(self.argument("paramFile"))
        trainer = get_trainer(params)
        progBar = self._get_progress_bar(trainer.nEpisodes)
        msg = f"<info>Episode Reward</info>: {trainer.episodeReward}"
        progBar.set_message(msg)
        trainer.pre_train()
        return (params, trainer, progBar)

    # -----
    # _train
    # -----
    def _train(
        self, params: DictConfig, trainer: BaseTrainer, progBar: ProgressBar
    ) -> Tuple:
        progBar.start()
        for trainer.episode in range(trainer.nEpisodes):
            trainer.train_step_start()
            trainer.train()
            msg = f"<info>Episode Reward</info>: {trainer.episodeReward}"
            progBar.set_message(msg)
            trainer.train_step_end()
            progBar.advance()
            if episode % params.io.checkpointFreq == 0:
                save_checkpoint(trainer, params)
        progBar.finish()
        return (params, trainer, progBar)

    # -----
    # _cleanup
    # -----
    def _cleanup(
        self, params: DictConfig, trainer: BaseTrainer, progBar: ProgressBar
    ) -> None:
        trainer.post_train()
        save_final_model(trainer, params.io.checkpointBase, params.io.outputDir)

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
