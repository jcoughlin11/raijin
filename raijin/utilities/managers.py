import gym

from raijin.utilities.register import registry


# ============================================
#                 get_trainer
# ============================================
def get_trainer(params):
    env = get_env(params.env.name)
    agent = registry[params.agent.name](params.agent)
    memory = registry[params.memory.name](params.memory) 
    pipeline = registry[params.pipeline.name](params.pipeline)
    nets = get_nets(params.nets, pipeline.traceLen, env.action_space.n)
    optimizers = get_optimizers(params.optimizers, nets)
    lossFunctions = get_loss_functions(params.losses)
    trainer = registry[params.trainer.name](agent, lossFunctions, memory, nets, optimizers, params.trainer, pipeline)
    return trainer


# ============================================
#                   get_env
# ============================================
def get_env(envName):
    env = gym.make(envName)
    return env


# ============================================
#                  get_nets
# ============================================
def get_nets(params, inChannels, nActions):
    nets = []
    for _, netParams in params.items():
        nets.append(registry[netParams.name](inChannels, nActions, netParams))
    return nets


# ============================================
#               get_optimizers
# ============================================
def get_optimizers(params, nets):
    opts = []
    for i, (_, optParams) in enumerate(params.items()):
        opts.append(registry[optParams.name](nets[i].parameters(), optParams))
    return opts


# ============================================
#             get_loss_functions
# ============================================
def get_loss_functions(params):
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
