import os
import subprocess


# ============================================
#               sanitize_path
# ============================================
def sanitize_path(path: str) -> str:
    path = os.path.expandvars(path)
    path = os.path.expanduser(path)
    return os.path.abspath(path)


# ============================================
#              get_numbered_dir
# ============================================
def get_numbered_dir(outputDir: str, base: str) -> str:
    outputDir = sanitize_path(outputDir)
    # Get the most recent checkpoint number
    dirNum = get_dir_num(outputDir, base)
    # Create new checkpoint directory
    nextDir = os.path.join(outputDir, f"{base}_{dirNum+1}")
    os.makedirs(nextDir)
    return nextDir


# ============================================
#                 get_dir_num
# ============================================
def get_dir_num(outputDir: str, base: str) -> int:
    if not os.path.isdir(outputDir):
        return -1
    dirs = []
    for c in os.listdir(outputDir):
        if c.startswith(f"{base}") and os.path.isdir(
            os.path.join(outputDir, c)
        ):
            dirs.append(c)
    if len(dirs) == 0:
        return -1
    dirNums = [int(c.split("_")[1]) for c in dirs]
    dirNums = sorted(dirNums, reverse=True)
    return dirNums[0]


# ============================================
#             package_iteration
# ============================================
def package_iteration(outputDir: str) -> None:
    """
    Moves any existing checkpoint directories to a `run_x` directory
    within `outputDir`.

    In order to calculate metric averages and variances, we need
    multiple runs of the exact same model. This allows each run to be
    packaged together within the same parent output directory, so we
    know that each run was of the same model.

    This also serves to make anaylsis easier, since we can loop over
    each run directory in a given `outputDir`.
    """
    outputDir = sanitize_path(outputDir)
    chkpts = []
    runs = []
    # Make sure there are checkpoints to move
    for obj in os.listdir(outputDir):
        if obj.startswith("checkpoint"):
            obj = os.path.join(outputDir, obj)
            if os.path.isdir(obj):
                chkpts.append(obj)
        elif obj.startswith("run"):
            if os.path.isdir(os.path.join(outputDir, obj)):
                runs.append(obj)
    if len(chkpts) == 0:
        return
    # Get the most recent run directory
    if len(runs) == 0:
        runNum = 0
    else:
        runNums = [int(r.split("_")[1]) for r in runs]
        runNums = sorted(runNums, reverse=True)
        runNum = runNums[0] + 1
    # Create the new run directory
    runDir = os.path.join(outputDir, f"run_{runNum}")
    os.mkdir(runDir)
    # Move checkpoints, params, and metric files
    for obj in os.listdir(outputDir):
        if not obj.startswith("run"):
            obj = os.path.join(outputDir, obj)
            subprocess.call(["mv", obj, runDir])
