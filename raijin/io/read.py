from omegaconf import OmegaConf as config

from raijin.utilities.io_utilities import sanitize_path


# ============================================
#             read_parameter_file
# ============================================
def read_parameter_file(paramFile):
    paramFile = sanitize_path(paramFile)
    params = config.load(paramFile)
    return params
