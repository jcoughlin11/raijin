import os
import pathlib

from gitinfo import get_git_info
import h5py
import numpy as np
from omegaconf import OmegaConf as config
from omegaconf.dictconfig import DictConfig
import torch
import yaml

from raijin import __version__ as raijinVersion
from raijin.memory import base_memory as bm
from raijin.trainers import base_trainer as bt
from raijin.utilities.io_utilities import get_numbered_dir
from raijin.utilities.io_utilities import sanitize_path


# ============================================
#               save_checkpoint
# ============================================
def save_checkpoint(trainer: "bt.BaseTrainer", params: DictConfig) -> None:
    """
    Saves the state of the trainer in a checkpoint directory.

    The checkpoint directories are numbered sequentially, e.g.,
    `checkpoint_0`, `checkpoint_1`, etc.

    Each checkpoint contains:
        * A copy of the parameter file
        * The metrics
        * The stateful parameters for:
            * The optimizer
            * The loss
            * The network(s)
            * The trainer
            * The pipeline
            * The agent
        * The memory buffer

    See: https://tinyurl.com/ycyuww2c
    """
    # Create checkpoint directory
    chkptDir = get_numbered_dir(params.io.outputDir, "checkpoint")
    # Save copy of parameter file
    save_params(chkptDir, params)
    # Save metrics
    trainer.metrics.save(chkptDir)
    # Save state dicts
    save_state_dicts(trainer, chkptDir, params.io.checkpointBase)
    # Save experience buffer
    save_memory(trainer.memory, chkptDir)
    # Save current codebase hash for reproducability
    save_version(chkptDir)


# ============================================
#                 save_params
# ============================================
def save_params(outputDir: str, params: DictConfig) -> None:
    outputDir = sanitize_path(outputDir)
    with open(os.path.join(outputDir, "params.yaml"), "w") as fd:
        yaml.safe_dump(config.to_yaml(params), fd)


# ============================================
#             save_state_dicts
# ============================================
def save_state_dicts(
    trainer: "bt.BaseTrainer", outputDir: str, baseName: str
) -> None:
    stateDict = trainer.state_dict()
    chkptFile = os.path.join(outputDir, f"{baseName}_state_dicts.tar")
    torch.save(stateDict, chkptFile)


# ============================================
#              save_final_model
# ============================================
def save_final_model(
    trainer: "bt.BaseTrainer", baseName: str, outputDir: str
) -> None:
    """
    Saves the network parameters once training is finished.

    See: https://tinyurl.com/hr7fw54w
    """
    outputDir = sanitize_path(outputDir)
    if not os.path.isdir(outputDir):
        os.makedirs(outputDir)
    modelFile = os.path.join(outputDir, "model.pt")
    torch.save(trainer.net.state_dict(), modelFile)
    # Save current codebase hash for reproducability
    save_version(outputDir)


# ============================================
#                save_memory
# ============================================
def save_memory(memory: "bm.BaseMemory", outputDir: str) -> None:
    """
    Saves the memory buffer.

    This is done by breaking each experience into its component parts
    (states, actions, rewards, nextStates, and dones) and then saving
    each one to is own hdf5 file.
    """
    outputDir = os.path.join(outputDir, "memory")
    if not os.path.exists(outputDir):
        os.mkdir(outputDir)
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
    statesShape = list(memory.buffer[0].state.numpy().shape) + [m,]
    statesDs = fs.create_dataset("states", statesShape, dtype=np.float32)
    actionsDs = fa.create_dataset("actions", m, dtype=np.int32)
    rewardsDs = fr.create_dataset("rewards", m, dtype=np.float32)
    nextStatesDs = fn.create_dataset(
        "nextStates", statesShape, dtype=np.float32
    )
    donesDs = fd.create_dataset("dones", m, dtype=np.int32)
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


# ============================================
#                save_version
# ============================================
def save_version(outputDir):
    """
    Gets the current git info and saves it to a `git_info.yaml` file.

    Includes:
        - current commit hash
        - path to the .git directory
        - commit message
        - tree hash
        - parent hash
        - commit author
        - commit date
    """
    gitInfo = get_git_info(dir=pathlib.Path(__file__).parent.absolute())
    with open(os.path.join(outputDir, "git_info.yaml"), "w") as fd:
        yaml.safe_dump(gitInfo, fd)


# ============================================
#               display_banner
# ============================================
def display_banner(display_function):
    """
    Prints code name, logo, and version information.
    """
    display_function("Raijin".center(os.get_terminal_size().columns))
    display_function(f"v{raijinVersion}".center(os.get_terminal_size().columns))
