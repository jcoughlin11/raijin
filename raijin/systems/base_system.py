from abc import ABC
from abc import abstractmethod

import pytorch_lightning as pl

from raijin.utilities.register import register_object


# ============================================
#                 BaseSystem
# ============================================
class BaseSystem(ABC, pl.LightningModule):
    # -----
    # subclass hook
    # -----
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        register_object(cls)

    # -----
    # constructor
    # -----
    def __init__(self):
        super().__init__()

    # -----
    # forward
    # -----
    @abstractmethod
    def forward(self):
        """
        The lightning docs recommend using this method ONLY for
        prediction. Takes whatever args and kwargs are needed for a
        forward pass.
        """
        pass

    # -----
    # configure_optimizers
    # -----
    @abstractmethod
    def configure_optimizers(self):
        """
        Sets up the optimizer(s). Can return:
            1. A single optimizer
            2. A list of optimizers
            3. Two lists: a list of optimizers and a list of learning rate schedulers
            4. A dictionary with an optimizer key and an optional "lr_scheduler" key
            5. A list of dictionaries
            6. None (Trainer.fit() does not use an optimizer)
        See: https://tinyurl.com/3wwhvcft
        """
        pass

    # -----
    # training_step
    # -----
    @abstractmethod
    def training_step(self):
        """
        Contains the full training loop. Used ONLY during training. Takes:
            batch (output of Dataloader)
            batch index (integer, used for logging)
            optimizer index (only needed if multiple optimizers are being used)
            hiddens: I don't think I need to worry about this.

        Returns: The loss tensor, a dictionary (that MUST contain a loss key), or None
        Can use forward.
        """
        pass

    # -----
    # learn
    # -----
    @abstractmethod
    def learn(self):
        """
        This isn't actually a lightning method. I'm just requiring it for myself.
        """
        pass
