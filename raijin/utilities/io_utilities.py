import os


# ============================================
#               sanitize_path
# ============================================
def sanitize_path(path):
    path = os.path.expandvars(path)
    path = os.path.expanduser(path)
    return os.path.abspath(path)


# ============================================
#               get_chkpt_num
# ============================================
def get_chkpt_num(outputDir):
    if not os.path.isdir(outputDir):
        return -1
    chkpts = []
    for c in os.listdir(outputDir):
        if c.startswith("checkpoint") and os.path.isdir(os.path.join(outputDir, c)):
            chkpts.append(c)
    chkpts = sorted(chkpts, reverse=True)
    if len(chkpts) == 0:
        return -1
    return int(chkpts[0].split("_")[1])
