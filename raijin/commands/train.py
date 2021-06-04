from datetime import datetime as dt

from cleo import Command
import gym

from raijin.io.read import read_parameter_file
from raijin.io.write import display_banner
from raijin.io.write import save_checkpoint
from raijin.io.write import save_final_model
from raijin.io.write import save_params
from raijin.metrics.metric_list import MetricList
from raijin.utilities.io_utilities import package_iteration
from raijin.utilities.managers import check_device
from raijin.utilities.managers import get_loss_functions
from raijin.utilities.managers import get_nets
from raijin.utilities.managers import get_optimizers
from raijin.utilities.performance import get_performance_stats
from raijin.utilities.register import registry


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
        display_banner(self.line)
        self.line("<warning>Initializing...</warning>")
        self._setup()
        self.line("<warning>Training...</warning>")
        self.line("\n")
        self._train()
        self.line("\n")
        self.line("<warning>Cleaning up...</warning>")
        self._cleanup()

    # -----
    # setup
    # -----
    def _setup(self) -> None:
        # Read parameter file
        self.params = read_parameter_file(self.argument("paramFile"))
        # If gpu is selected, make sure we have cuda. Otherwise, use
        # a cpu
        self.params.device.name = check_device(self.params.device.name)
        # Get the progress bar
        self._get_progress_bar(self.params.trainer.nEpisodes)
        # If iterating and there are existing checkpoints from a
        # previous run then we need to move those so checkpoints
        # aren't overridden
        if self.option("iterate"):
            package_iteration(self.params.io.outputDir)
        # Get the trainer
        self._get_trainer()
        # Print the parameters being used
        self._print_params()

    # -----
    # _train
    # -----
    def _train(self) -> None:
        self.progBar.start()
        # Run the pre-training hook
        self.trainer.pre_train()
        # Loop over the desired number of episodes
        for self.trainer.episode in range(self.trainer.nEpisodes):
            # Run the pre-training step hook
            self.trainer.episode_start()
            # Main body of training loop
            self.trainer.train_episode()
            msg = ""
            performanceStats = get_performance_stats(self.params.io.outputDir)
            for statName, statValue in performanceStats.items():
                msg += f"\n\t<info>{statName:<14}</info> : {statValue}%"
            s = "Episode Reward"
            msg = f"\n\t<info>{s:<14}</info> : {self.trainer.episodeReward}"
            self.progBar.set_message(msg)
            # End training step hook
            self.trainer.episode_end()
            self.progBar.advance()
            # Save, if applicable
            if self.trainer.episode % self.params.io.checkpointFreq == 0:
                save_checkpoint(self.trainer, self.params)
        # Run the post-training hook
        self.trainer.post_train()
        self.progBar.finish()

    # -----
    # _cleanup
    # -----
    def _cleanup(self) -> None:
        # In case there aren't any checkpoints, we save a copy of the
        # parameter file
        save_params(self.params.io.outputDir, self.params)
        # Save the trained model
        save_final_model(
            self.trainer,
            self.params.io.checkpointBase,
            self.params.io.outputDir,
        )
        # Move all existing checkpoint directories to a run_x directory,
        # if applicable
        if self.option("iterate"):
            package_iteration(self.params.io.outputDir)
        # Print completion time
        date = dt.now().strftime("%b %d, %Y; %H:%M")
        self.line(f"<warning>Completed at</warning>: {date}")

    # -----
    # _get_trainer
    # -----
    def _get_trainer(self) -> None:
        device = self.params.device.name
        env = gym.make(self.params.env.name)
        pipeline = registry[self.params.pipeline.name](self.params.pipeline)
        agent = registry[self.params.agent.name](
            env, pipeline, self.params.agent, device
        )
        memory = registry[self.params.memory.name](self.params.memory, device)
        nets = get_nets(self.params.nets, pipeline.traceLen, env.action_space.n)
        optimizers = get_optimizers(self.params.optimizers, nets)
        lossFunctions = get_loss_functions(self.params.losses)
        metrics = MetricList([registry[m]() for m in self.params.metrics])
        self.trainer = registry[self.params.trainer.name](
            agent,
            lossFunctions,
            memory,
            nets,
            optimizers,
            self.params.trainer,
            device,
            metrics
        )

    # -----
    # _get_progress_bar
    # -----
    def _get_progress_bar(self, nEpisodes: int) -> None:
        performanceStats = get_performance_stats(self.params.io.outputDir)
        self.progBar = self.progress_bar(nEpisodes)
        s1 = "Episode"
        s2 = "Elsapsed Time"
        formatStr = f"\t<info>{s1:<14}</info> : %current%/%max%"
        formatStr += f"\n\t<info>{s2:<14}</info> : %elapsed%"
        formatStr += "\n\t%message%"
        for statName, statValue in performanceStats.items():
            formatStr += f"\n\t<info>{statName:<14}</info> : {statValue}%"
        s = "Episode Reward"
        formatStr += f"\n\t<info>{s:<14}</info> : {0.0}"
        self.progBar.set_format(formatStr)

    # -----
    # _print_params
    # -----
    def _print_params(self) -> None:
        pairs = [
            ("Running on", self.params.device.name),
            ("Trainer", self.params.trainer.name),
            ("Game", self.params.env.name),
            ("Memory", self.params.memory.name),
            ("Pipeline", self.params.pipeline.name),
            ("Agent", self.params.agent.name),
        ]
        z = zip(
            self.params.nets.keys(),
            self.params.optimizers.keys(),
            self.params.losses.keys(),
        )
        for i, (net, opt, loss) in enumerate(z):
            pairs.append((f"Network {i+1}", self.params.nets[net]["name"]))
            pairs.append(
                (f"Optimizer {i+1}", self.params.optimizers[opt]["name"])
            )
            pairs.append((f"Loss {i+1}", self.params.losses[loss]["name"]))
        pairs.append(("Output directory", self.params.io.outputDir))
        msg = "\n\t"
        for (s, p) in pairs:
            msg += f"<info>{s:<16}</info> : {p}\n\t"
        self.line(msg)
