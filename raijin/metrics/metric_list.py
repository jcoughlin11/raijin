# ============================================
#                 MetricList
# ============================================
class MetricList:
    """
    Interface to act on multiple metrics at once.
    """

    __name__ = "MetricList"

    # -----
    # constructor
    # -----
    def __init__(self, metrics):
        self.metrics = metrics

    # -----
    # reset
    # -----
    def reset(self) -> None:
        for m in self.metrics:
            m.reset()

    # -----
    # update
    # -----
    def update(self, *args, **kwargs) -> None:
        for m in self.metrics:
            m.update(*args, **kwargs)

    # -----
    # log
    # -----
    def log(self) -> None:
        for m in self.metrics:
            m.log()
