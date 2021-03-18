import warnings

import pretty_errors

from .app import RaijinApplication


warnings.filterwarnings("ignore")


# ============================================
#                     run
# ============================================
def run():
    RaijinApplication().run()
    print("Done.")
