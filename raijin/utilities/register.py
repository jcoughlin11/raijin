import torch


registry = {}


lossRegister = {
    "BCELoss": torch.nn.BCELoss,
    "BCEWithLogitsLoss": torch.nn.BCEWithLogitsLoss,
    "CosineEmbeddingLoss": torch.nn.CosineEmbeddingLoss,
    "CrossEntropyLoss": torch.nn.CrossEntropyLoss,
    "CTCLoss": torch.nn.CTCLoss,
    "HingeEmbeddingLoss": torch.nn.HingeEmbeddingLoss,
    "KLDivLoss": torch.nn.KLDivLoss,
    "L1Loss": torch.nn.L1Loss,
    "MarginRankingLoss": torch.nn.MarginRankingLoss,
    "MSELoss": torch.nn.MSELoss,
    "MultiLabelMarginLoss": torch.nn.MultiLabelMarginLoss,
    "MultiLabelSoftMarginLoss": torch.nn.MultiLabelSoftMarginLoss,
    "MultiMarginLoss": torch.nn.MultiMarginLoss,
    "NLLLoss": torch.nn.NLLLoss,
    "NLLLoss2d": torch.nn.NLLLoss2d,
    "PoissonNLLLoss": torch.nn.PoissonNLLLoss,
    "SmoothL1Loss": torch.nn.SmoothL1Loss,
    "SoftMarginLoss": torch.nn.SoftMarginLoss,
    "TripletMarginLoss": torch.nn.TripletMarginLoss,
    "TripletMarginWithDistanceLoss": torch.nn.TripletMarginWithDistanceLoss,
}
registry.update(lossRegister)


optimizerRegister = {
    "Adadelta": torch.optim.Adadelta,
    "Adagrad": torch.optim.Adagrad,
    "Adam": torch.optim.Adam,
    "Adamax": torch.optim.Adamax,
    "AdamW": torch.optim.AdamW,
    "ASGD": torch.optim.ASGD,
    "LBFGS": torch.optim.LBFGS,
    "RMSprop": torch.optim.RMSprop,
    "Rprop": torch.optim.Rprop,
    "SGD": torch.optim.SGD,
    "SparseAdam": torch.optim.SparseAdam,
}
registry.update(optimizerRegister)


# ============================================
#               register_object
# ============================================
def register_object(cls) -> None:
    registry[cls.__name__] = cls
