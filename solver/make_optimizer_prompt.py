import torch


def make_optimizer_prompt(model,cfg):
    params = []
    keys = []
    for key, value in model.named_parameters():
        if "clsctx" in key:
            lr = cfg.prompt_lr
            weight_decay = 1e-4
            params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]
            keys += [key]

        if 'dcgrl' in key:
            lr = cfg.lamda
            weight_decay = 1e-4
            params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]
            keys += [key]

    if cfg.optimizer == 'SGD':
        optimizer = getattr(torch.optim, cfg.optimizer)(params, momentum=0.95)
    elif cfg.optimizer == 'AdamW':
        optimizer = torch.optim.AdamW(params, lr=cfg.prompt_lr, weight_decay=1e-4)
    else:
        optimizer = getattr(torch.optim, 'Adam')(params)
    return optimizer

def make_optimizer_prompt_domain(model,cfg):
    params = []
    keys = []
    for key, value in model.named_parameters():
        if "dmctx" in key:
            lr = cfg.prompt_lr
            weight_decay = 1e-4
            params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]
            keys += [key]

        if "dc" in key and 'dcgrl' not in key:
            print(key)
            lr = cfg.lamda
            weight_decay = 1e-4
            params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]
            keys += [key]

    if cfg.optimizer == 'SGD':
        optimizer = getattr(torch.optim, cfg.optimizer)(params, momentum=0.95)
    elif cfg.optimizer == 'AdamW':
        optimizer = torch.optim.AdamW(params, lr=cfg.prompt_lr, weight_decay=1e-4)
    else:
        optimizer = getattr(torch.optim, 'Adam')(params)
    return optimizer


def make_optimizer_for_IE(model,cfg):
    params = []
    keys = []
    for key, value in model.named_parameters():
        if "text_encoder" in key:
            value.requires_grad_(False)
            continue
        if "prompt_learner" in key:
            value.requires_grad_(False)
            continue
        if not value.requires_grad:
            continue
        lr = cfg.image_encoder_lr
        weight_decay = 0.0001
        if "bias" in key:
            lr = cfg.image_encoder_lr * 2
        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]
        keys += [key]
    if cfg.optimizer == 'SGD':
        optimizer = getattr(torch.optim, cfg.optimizer)(params, momentum=0.95)
    elif cfg.optimizer == 'AdamW':
        optimizer = torch.optim.AdamW(params, lr=cfg.image_encoder_lr, weight_decay=1e-4)
    else:
        optimizer = getattr(torch.optim, 'Adam')(params)

    return optimizer
