from collections import namedtuple


Experience = namedtuple(
    "Experience",
    [
        "state",
        "action",
        "reward",
        "nextState",
        "done",
    ]
)
