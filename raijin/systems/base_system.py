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
        pass

    # -----
    # configure_optimizers
    # -----
    @abstractmethod
    def configure_optimizers(self):
        pass

    # -----
    # training_step
    # -----
    @abstractmethod
    def training_step(self):
        pass
