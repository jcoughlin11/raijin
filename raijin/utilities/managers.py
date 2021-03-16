import gym

from raijin.utilities.register import registry


# ============================================
#                 get_trainer
# ============================================
def get_trainer(params):
    env = get_env(params.env.name)
    memory = registry[params.memory.name](params.memory)
    nets = get_component(params.nets) 
    pipeline = registry[params.pipeline.name](params.pipeline) 
    optimizers = get_component(params.optimizers)
    losses = get_component(params.losses)
    agent = registry[params.agent.name](params.agent)
    return trainer


# ============================================
#                   get_env
# ============================================
def get_env(envName):
    env = gym.make(envName)
    return env


# ============================================
#               get_component
# ============================================
def get_component(params):
    components = []
    for componentName, componentParams in params.items():
        components.append(registry[componentName](componentParams))
    return components
