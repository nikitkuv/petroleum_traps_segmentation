import torch


def get_optimizer(model):
    enc, dec = [], []

    for name, param in model.named_parameters():
        if "encoder" in name:
            enc.append(param)
        else:
            dec.append(param)

    return torch.optim.AdamW([
        {"params": enc, "lr": 1e-5},
        {"params": dec, "lr": 1e-4},
    ], weight_decay=1e-4)


def get_scheduler(optimizer):
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=8, factor=0.5
    )