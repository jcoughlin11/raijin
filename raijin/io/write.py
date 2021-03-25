import os

import torch
import yaml

from raijin.utilities.io_utilities import get_chkpt_num
from raijin.utilities.io_utilities import sanitize_path
from raijin.utilities.managers import get_state_dicts


# ============================================
#               save_checkpoint
# ============================================
def save_checkpoint(trainer, episodeNum, params):
    # https://tinyurl.com/ycyuww2c
    outputDir = sanitize_path(params.io.outputDir)
    # Get the most recent checkpoint number
    chkptNum = get_chkpt_num(outputDir)
    # Create new checkpoint directory
    chkptDir = os.path.join(outputDir, f"checkpoint_{chkptNum+1}")
    os.makedirs(chkptDir)
    # Get state dictionaries
    stateDicts = get_state_dicts(trainer)
    # Save copy of parameter file
    stateDicts["episodeNum"] = episodeNum
    chkptFile = os.path.join(chkptDir, f"{params.io.checkpointBase}.tar")
    torch.save(stateDicts, chkptFile)
    # Save copy of parameter file
    with open(os.path.join(chkptDir, "params.yaml"), "w") as fd:
        yaml.safe_dump(params, fd)
    # Save experience buffer
    save_memory(trainer.memory, chkptDir)


# ============================================
#              save_final_model
# ============================================
def save_final_model(trainer, baseName, outputDir):
    # https://tinyurl.com/hr7fw54w
    outputDir = sanitize_path(params.io.outputDir)
    if not os.path.isdir(outputDir):
        os.makedirs(outputDir)
    modelFile = os.path.join(outputDir, "model.pt")
    torch.save(trainer.net.state_dict(), modelFile)


# ============================================
#                save_memory
# ============================================
def save_memory(memory, outputDir):
    # Brute force right now. There's probably a more efficient way
    # to do this
    statesFile = os.path.join(outputDir, "buffer_states.h5py")
    actionsFile = os.path.join(outputDir, "buffer_actions.h5py")
    rewardsFile = os.path.join(outputDir, "buffer_rewards.h5py")
    nextStatesFile = os.path.join(outputDir, "buffer_nextStates.h5py")
    donesFile = os.path.join(outputDir, "buffer_dones.h5py")
    fs = h5py.File(statesFile, "w")
    fa = h5py.File(actionsFile, "w")
    fr = h5py.File(rewardsFile, "w")
    fn = h5py.File(nextStatesFile, "w")
    fd = h5py.File(donesFile, "w")
    m = len(memory)
    statesShape = list(memory[0].state.to_numpy().shape) + N
    statesDs = fs.create_dataset("states", statesShape, dtype=np.float)
    actionsDs = fa.create_dataset("actions", N, dtype=np.int)
    rewardsDs = fr.create_dataset("rewards", N, dtype=np.float)
    nextStatesDs = fn.create_dataset("nextStates", statesShape, dtype=np.float)
    donesDs = fd.create_dataset("dones", N, dtype=np.int)
    for i, experience in enumerate(memory):
        statesDs[:,:,i] = experience.state.to_numpy()
        actionsDs[i] = experience.action
        rewardsDs[i] = experience.reward
        nextStatesDs[:,:,i] = experience.nextState
        donesDs[i] = experience.done