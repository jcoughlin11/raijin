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
        params = read_parameter_file(self.argument("paramFile"))
        trainer = get_trainer(params)
        trainer.train()
