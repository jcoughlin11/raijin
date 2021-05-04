from typing import Union

from omegaconf import OmegaConf as config
from omegaconf.dictconfig import DictConfig
from omegaconf.listconfig import ListConfig

from raijin.utilities.io_utilities import sanitize_path


# ============================================
#             read_parameter_file
# ============================================
def read_parameter_file(paramFile: str) -> Union[DictConfig, ListConfig]:
    paramFile = sanitize_path(paramFile)
    params = config.load(paramFile)
    return params
