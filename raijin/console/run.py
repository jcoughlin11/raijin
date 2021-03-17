import pretty_errors

from .app import RaijinApplication


# ============================================
#                     run
# ============================================
def run():
    RaijinApplication().run()
    print("Done.")
