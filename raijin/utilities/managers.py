from typing import List

from omegaconf.dictconfig import DictConfig
import torch

from raijin.utilities.register import registry


# ============================================
#                  get_nets
# ============================================
def get_nets(params: DictConfig, inChannels: int, nActions: int) -> List:
    nets = []
    for _, netParams in params.items():
        nets.append(
            registry[netParams.name](inChannels, nActions, params=netParams)
        )
    return nets


# ============================================
#               get_optimizers
# ============================================
def get_optimizers(params: DictConfig, nets: List) -> List:
    opts = []
    for i, (_, optParams) in enumerate(params.items()):
        cls = registry[optParams.name]
        del optParams["name"]
        opts.append(cls(nets[i].parameters(), **optParams))
    return opts


# ============================================
#             get_loss_functions
# ============================================
def get_loss_functions(params: DictConfig) -> List:
    lossFunctions = []
    for _, lossParams in params.items():
        # If name is the only parameter
        if len(lossParams) == 1:
            lossFunctions.append(registry[lossParams.name]())
        else:
            cls = registry[lossParams.name]
            del lossParams.name
            lossFunctions.append(cls(lossParams))
    return lossFunctions


# ============================================
#                check_device
# ============================================
def check_device(device: str) -> str:
    # If a gpu is chosen but there isn't any cuda support, switch
    # to a cpu
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    return device
