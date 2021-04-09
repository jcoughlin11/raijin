from copy import deepcopy
from typing import List

from omegaconf.dictconfig import DictConfig
import torch

from .qtrainer import QTrainer


# ============================================
#                FixedQTrainer
# ============================================
class FixedQTrainer(QTrainer):
    """
    Learning method from [Lillicrap et al. 2016][1].
    Training loop from [Mnih et al. 2013][2].

    [1]: https://arxiv.org/abs/1509.02971v6
    [2]: https://arxiv.org/abs/1312.5602
    """
    __name__ = "FixedQTrainer"

    # -----
    # constructor
    # -----
    def __init__(
        self,
        agent: "ba.BaseAgent",
        lossFunctions: List,
        memory: "bm.BaseMemory",
        nets: List,
        optimizers: List,
        params: DictConfig,
        device: str
    ) -> None:
        super().__init__(agent, lossFunctions, memory, nets, optimizers, params, device)
        # How frequently to update the target network
        self.updateFreq = params.updateFreq
        # Set up the target network
        self.targetNet = deepcopy(self.net)
        # Not sure if these two steps are needed since net will have already
        # done them in QTrainer's __init__. Are they preserved by the copy?
        self.targetNet.to(self.device)
        self.targetNet.train()

    # -----
    # train_step_start
    # -----
    def train_step_start(self) -> None:
        # Update target network, if needed
        if self.episode % self.updateFreq == 0:
            print("Updating target network.")
            self.targetNet.load_state_dict(self.net.state_dict())

    # -----
    # _get_targets
    # -----
    def _get_targets(
        self,
        nextStates: torch.Tensor,
        dones: torch.Tensor,
        rewards: torch.Tensor,
    ) -> torch.Tensor:
        """
        In DQL the targets (labels) are determined from the same
        network that they are being used to update. As such, there can
        be a lot of noise due to values constantly jumping wildly. This
        affects the speed of convergence.

        In fixed-Q, a second network, called the target network, is used
        to determine the target values. The weights in the primary
        network are copied over to the target network every N steps.
        This reduces the amount of jumping around in the target network,
        which makes its predicted labels more stable and helps speed up
        convergence.
        """
        # qNext should have shape (N, nActions)
        qNext = self.targetNet(nextStates)
        # the max operation doesn't return a tensor; it returns an object
        # that contains both the values and indices
        # qNextMax should have shape (N, 1)
        qNextMax = torch.max(qNext, 1, keepdims=True).values
        maskedVals = (1.0 - dones) * qNextMax
        targets = rewards + self.discountRate * maskedVals
        return targets
