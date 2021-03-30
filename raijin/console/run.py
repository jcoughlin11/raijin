import warnings

import pretty_errors  # noqa: F401

from .app import RaijinApplication


warnings.filterwarnings("ignore")


# ============================================
#                     run
# ============================================
def run() -> None:
    RaijinApplication().run()
