import torch


def make_optimizer(args, model):
    params = []
    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        lr = args.lr
        weight_decay = args.weight_decay
        # if "bias" in key:
        #     lr = cfg.SOLVER.BASE_LR * cfg.SOLVER.BIAS_LR_FACTOR
        #     weight_decay = cfg.SOLVER.WEIGHT_DECAY_BIAS

        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]


    optimizer = getattr(torch.optim, "SGD")(params, momentum=0.9)

        # optimizer = torch.optim.AdamW(params, lr=cfg.SOLVER.BASE_LR, weight_decay=cfg.SOLVER.WEIGHT_DECAY)
        #
        # optimizer = getattr(torch.optim, cfg.SOLVER.OPTIMIZER_NAME)(params)

    return optimizer
