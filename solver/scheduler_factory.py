""" Scheduler Factory
Hacked together by / Copyright 2020 Ross Wightman
"""
from .cosine_lr import CosineLRScheduler


def create_scheduler(epoch,lr, optimizer):
    num_epochs = epoch
    # type 1
    # lr_min = 0.01 * cfg.lr
    # warmup_lr_init = 0.001 * cfg.lr
    # type 2
    # lr_min = 0.002 * lr
    # warmup_lr_init = 0.01 * lr
    # type 3
    # lr_min = 0.001 * cfg.SOLVER.BASE_LR
    # warmup_lr_init = 0.01 * cfg.SOLVER.BASE_LR
    lr_min = 1e-6
    warmup_lr_init = 0.00001
    warmup_t = 5
    noise_range = None

    lr_scheduler = CosineLRScheduler(
        optimizer,
        t_initial=num_epochs,
        lr_min=lr_min,
        t_mul=1.,
        decay_rate=0.1,
        warmup_lr_init=warmup_lr_init,
        warmup_t=warmup_t,
        cycle_limit=1,
        t_in_epochs=True,
        noise_range_t=noise_range,
        noise_pct=0.67,
        noise_std=1.,
        noise_seed=42,
    )
    return lr_scheduler
