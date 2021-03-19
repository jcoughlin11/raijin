import gym

from raijin.utilities.register import registry


# ============================================
#                 get_trainer
# ============================================
def get_trainer(params):
    env = get_env(params.env.name)
    pipeline = registry[params.pipeline.name](params.pipeline)
    agent = registry[params.agent.name](env, pipeline, params.agent)
    memory = registry[params.memory.name](params.memory) 
    nets = get_nets(params.nets, pipeline.traceLen, env.action_space.n)
    optimizers = get_optimizers(params.optimizers, nets)
    lossFunctions = get_loss_functions(params.losses)
    trainer = registry[params.trainer.name](agent, lossFunctions, memory, nets, optimizers, params.trainer)
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
        nets.append(registry[netParams.name](inChannels, nActions, params=netParams))
    return nets


# ============================================
#               get_optimizers
# ============================================
def get_optimizers(params, nets):
    opts = []
    for i, (_, optParams) in enumerate(params.items()):
        cls = registry[optParams.name]
        del optParams["name"]
        opts.append(cls(nets[i].parameters(), **optParams))
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


# ============================================
#               get_callbacks
# ============================================
def get_callbacks(params):
    callbacks = {}
    for callbackName, callback in params.items():
        if callback == "None":
            callback = None
        callbacks[callbackName] = callback
    return callbacks
