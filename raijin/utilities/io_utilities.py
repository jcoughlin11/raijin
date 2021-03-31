import os


# ============================================
#               sanitize_path
# ============================================
def sanitize_path(path: str) -> str:
    path = os.path.expandvars(path)
    path = os.path.expanduser(path)
    return os.path.abspath(path)


# ============================================
#               get_chkpt_dir
# ============================================
def get_chkpt_dir(outputDir: str) -> str:
    outputDir = sanitize_path(outputDir)
    # Get the most recent checkpoint number
    chkptNum = get_chkpt_num(outputDir)
    # Create new checkpoint directory
    chkptDir = os.path.join(outputDir, f"checkpoint_{chkptNum+1}")
    os.makedirs(chkptDir)
    return chkptDir


# ============================================
#               get_chkpt_num
# ============================================
def get_chkpt_num(outputDir: str) -> int:
    if not os.path.isdir(outputDir):
        return -1
    chkpts = []
    for c in os.listdir(outputDir):
        if c.startswith("checkpoint") and os.path.isdir(
            os.path.join(outputDir, c)
        ):
            chkpts.append(c)
    if len(chkpts) == 0:
        return -1
    chkptNums = [int(c.split("_")[1]) for c in chkpts]
    chkptNums = sorted(chkpts, reverse=True)
    return chkptNums[0]
