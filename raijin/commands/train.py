from datetime import datetime as dt
from typing import Tuple

from cleo import Command
from clikit.ui.components.progress_bar import ProgressBar
from omegaconf.dictconfig import DictConfig

from raijin.io.read import read_parameter_file
from raijin.io.write import save_checkpoint
from raijin.io.write import save_final_model
from raijin.io.write import save_params
from raijin.trainers.base_trainer import BaseTrainer
from raijin.utilities.io_utilities import package_iteration
from raijin.utilities.managers import check_device
from raijin.utilities.managers import get_trainer


# ============================================
#                TrainCommand
# ============================================
class TrainCommand(Command):
    """
    Trains an agent according to the given parameter file.

    train
        {paramFile : Yaml file containing run parameters.}
        {--i|iterate : Tells raijin that this is a repeated run for
            error calculation purposes. Any existing checkpoint
            directories will be packaged into a `run_1` directory and
            subsequent runs will be packaged accordingly, too.}
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
        self._cleanup(params, trainer)
        date = dt.now().strftime("%b %d, %Y; %H:%M")
        self.line(f"<warning>Completed at</warning>: {date}")

    # -----
    # _initialize
    # -----
    def _initialize(self) -> Tuple:
        params = read_parameter_file(self.argument("paramFile"))
        # If there are existing checkpoints from a previous run
        # because user maybe didn't know they were going to iterate,
        # then we need to move those before starting training on the
        # new iteration
        if self.option("iterate"):
            package_iteration(params.io.outputDir)
        # If gpu is selected, make sure we have cuda. Otherwise, use
        # a cpu
        params.device.name = check_device(params.device.name)
        self._print_params(params)
        trainer = get_trainer(params)
        progBar = self._get_progress_bar(trainer.nEpisodes)
        s = "Episode Reward"
        msg = f"<info>{s:<14}</info> : {trainer.episodeReward}"
        progBar.set_message(msg)
        trainer.pre_train()
        return (params, trainer, progBar)

    # -----
    # _print_params
    # -----
    def _print_params(self, params):
        pairs = [
            ("Running on", params.device.name),
            ("Trainer", params.trainer.name),
            ("Game", params.env.name),
            ("Memory", params.memory.name),
            ("Pipeline", params.pipeline.name),
            ("Agent", params.agent.name),
        ]
        z = zip(params.nets.keys(), params.optimizers.keys(), params.losses.keys())
        for i, (net, opt, loss) in enumerate(z):
            pairs.append((f"Network {i+1}", params.nets[net]["name"]))
            pairs.append((f"Optimizer {i+1}", params.optimizers[opt]["name"]))
            pairs.append((f"Loss {i+1}", params.losses[loss]["name"]))
        pairs.append(("Output directory", params.io.outputDir))
        msg = "\n\t"
        for (s, p) in pairs:
            msg += f"<info>{s:<16}</info> : {p}\n\t"
        self.line(msg)

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
            s = "Episode Reward"
            msg = f"<info>{s:<14}</info> : {trainer.episodeReward}"
            progBar.set_message(msg)
            trainer.train_step_end()
            progBar.advance()
            if trainer.episode % params.io.checkpointFreq == 0:
                save_checkpoint(trainer, params)
        progBar.finish()
        return (params, trainer, progBar)

    # -----
    # _cleanup
    # -----
    def _cleanup(
        self, params: DictConfig, trainer: BaseTrainer) -> None:
        trainer.post_train()
        # In case there aren't any checkpoints, we save a copy of the
        # parameter file to be used during testing
        save_params(params.io.outputDir, params)
        save_final_model(trainer, params.io.checkpointBase, params.io.outputDir)
        # Move all existing checkpoint directories to a run_x directory
        if self.option("iterate"):
            package_iteration(params.io.outputDir)

    # -----
    # _get_progress_bar
    # -----
    def _get_progress_bar(self, nEpisodes: int) -> ProgressBar:
        progBar = self.progress_bar(nEpisodes)
        s1 = "Episode"
        s2 = "Elsapsed Time"
        formatStr = f"\t<info>{s1:<14}</info> : %current%/%max%"
        formatStr += f"\n\t<info>{s2:<14}</info> : %elapsed%"
        formatStr += "\n\t%message%"
        progBar.set_format(formatStr)
        return progBar
