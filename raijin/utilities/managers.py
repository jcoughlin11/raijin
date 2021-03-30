from typing import List

import gym
from omegaconf.dictconfig import DictConfig

from raijin.trainers.base_trainer import BaseTrainer
from raijin.utilities.register import registry


# ============================================
#                 get_trainer
# ============================================
def get_trainer(params: DictConfig) -> BaseTrainer:
    env = get_env(params.env.name)
    pipeline = registry[params.pipeline.name](params.pipeline)
    agent = registry[params.agent.name](env, pipeline, params.agent)
    memory = registry[params.memory.name](params.memory)
    nets = get_nets(params.nets, pipeline.traceLen, env.action_space.n)
    optimizers = get_optimizers(params.optimizers, nets)
    lossFunctions = get_loss_functions(params.losses)
    trainer = registry[params.trainer.name](
        agent, lossFunctions, memory, nets, optimizers, params.trainer
    )
    return trainer


# ============================================
#                   get_env
# ============================================
def get_env(envName: str) -> gym.Env:
    env = gym.make(envName)
    return env


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
#              get_state_dicts
# ============================================
def get_state_dicts(manager: BaseTrainer) -> dict:
    stateDicts = {}
    for attrVal in manager.__dict__.values():
        if hasattr(attrVal, "state_dict"):
            stateDict = attrVal.state_dict()
            # The loss and optimizer don't have __name__ attrs, but
            # the loss has a _get_name method. For the optimizer, we
            # have to use str(). This prints the parameters, too,
            # though, which need to be removed
            if hasattr(attrVal, "__name__"):
                name = attrVal.__name__
            elif hasattr(attrVal, "_get_name()"):
                name = attrVal._get_name()
            else:
                name = str(attrVal).split()[0]
            stateDicts[name] = stateDict
    if hasattr(manager, "state_dict"):
        stateDicts[manager.__name__] = manager.state_dict()
    return stateDicts
