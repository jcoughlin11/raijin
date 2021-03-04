from cleo import Command
import pytorch_lightning as pl

from raijin.io.read import read_parameter_file
from raijin.systems.system_utilities import get_system


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
        params = read_parameter_file(self.argument("paramFile"))
        system = get_system(params)
        trainer = pl.Trainer(**params.training)
        trainer.fit(system)
