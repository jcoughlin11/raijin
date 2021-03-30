import os

import h5py
import numpy as np
from omegaconf import OmegaConf as config
from omegaconf.dictconfig import DictConfig
import torch
import yaml

from raijin.memory.base_memory import BaseMemory
from raijin.trainers.base_trainer import BaseTrainer
from raijin.utilities.io_utilities import get_chkpt_num
from raijin.utilities.io_utilities import sanitize_path
from raijin.utilities.managers import get_state_dicts


# ============================================
#               save_checkpoint
# ============================================
def save_checkpoint(trainer: BaseTrainer, episodeNum: int, params: DictConfig) -> None:
    """
    Saves the state of the trainer in a checkpoint directory.

    The checkpoint directories are numbered sequentially, e.g.,
    `checkpoint_0`, `checkpoint_1`, etc.

    Each checkpoint contains:
        * A copy of the parameter file
        * The memory buffer
        * The stateful parameters for:
            * The optimizer
            * The loss
            * The network(s)
            * The trainer
            * The pipeline
            * The agent

    See: https://tinyurl.com/ycyuww2c
    """
    outputDir = sanitize_path(params.io.outputDir)
    # Get the most recent checkpoint number
    chkptNum = get_chkpt_num(outputDir)
    # Create new checkpoint directory
    chkptDir = os.path.join(outputDir, f"checkpoint_{chkptNum+1}")
    os.makedirs(chkptDir)
    # Get state dictionaries and save
    stateDicts = get_state_dicts(trainer)
    stateDicts["episodeNum"] = episodeNum
    chkptFile = os.path.join(chkptDir, f"{params.io.checkpointBase}.tar")
    torch.save(stateDicts, chkptFile)
    # Save copy of parameter file
    with open(os.path.join(chkptDir, "params.yaml"), "w") as fd:
        yaml.safe_dump(config.to_yaml(params), fd)
    # Save experience buffer
    save_memory(trainer.memory, chkptDir)


# ============================================
#              save_final_model
# ============================================
def save_final_model(trainer: BaseTrainer, baseName: str, outputDir: str) -> None:
    """
    Saves the network parameters once training is finished.

    See: https://tinyurl.com/hr7fw54w
    """
    outputDir = sanitize_path(outputDir)
    if not os.path.isdir(outputDir):
        os.makedirs(outputDir)
    modelFile = os.path.join(outputDir, "model.pt")
    torch.save(trainer.net.state_dict(), modelFile)


# ============================================
#                save_memory
# ============================================
def save_memory(memory: BaseMemory, outputDir: str) -> None:
    """
    Saves the memory buffer.

    This is done by breaking each experience into its component parts
    (states, actions, rewards, nextStates, and dones) and then saving
    each one to is own hdf5 file.
    """
    # Set up file names
    statesFile = os.path.join(outputDir, "buffer_states.h5py")
    actionsFile = os.path.join(outputDir, "buffer_actions.h5py")
    rewardsFile = os.path.join(outputDir, "buffer_rewards.h5py")
    nextStatesFile = os.path.join(outputDir, "buffer_nextStates.h5py")
    donesFile = os.path.join(outputDir, "buffer_dones.h5py")
    # Create file objects
    fs = h5py.File(statesFile, "w")
    fa = h5py.File(actionsFile, "w")
    fr = h5py.File(rewardsFile, "w")
    fn = h5py.File(nextStatesFile, "w")
    fd = h5py.File(donesFile, "w")
    # Initialize datasets
    m = len(memory.buffer)
    statesShape = list(memory.buffer[0].state.numpy().shape) + [
        m,
    ]
    statesDs = fs.create_dataset("states", statesShape, dtype=np.float)
    actionsDs = fa.create_dataset("actions", m, dtype=np.int)
    rewardsDs = fr.create_dataset("rewards", m, dtype=np.float)
    nextStatesDs = fn.create_dataset("nextStates", statesShape, dtype=np.float)
    donesDs = fd.create_dataset("dones", m, dtype=np.int)
    # Write data
    for i, experience in enumerate(memory.buffer):
        statesDs[:, :, :, i] = experience.state.numpy()
        actionsDs[i] = experience.action
        rewardsDs[i] = experience.reward
        nextStatesDs[:, :, :, i] = experience.nextState.numpy()
        donesDs[i] = experience.done
    # Close files
    fs.close()
    fa.close()
    fr.close()
    fn.close()
    fd.close()
