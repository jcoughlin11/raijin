import gym

from raijin.utilities.register import registry


# ============================================
#                 get_module
# ============================================
def get_system(params):
    env = get_env(params.env)
    memory = registry[params.memory.name](params.memory)
    nets = registry[params.network.name](params.network)
    agent = registry[params.agent.name](env, memory)
    dataset = registry[params.dataset.name](params.dataset)
    return registry[params.system.name](params.hyperParams, nets, agent, dataset)


# ============================================
#                   get_env
# ============================================
def get_env(envParams):
    env = gym.make(envParams.name)
    return env
