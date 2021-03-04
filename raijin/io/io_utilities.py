import os


# ============================================
#               sanitize_path
# ============================================
def sanitize_path(path):
    path = os.path.expandvars(path)
    path = os.path.expanduser(path)
    return os.path.abspath(path)
