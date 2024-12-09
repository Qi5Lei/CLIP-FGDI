from . import trans as T
from torch import nn


def bulid_transforms(cfg, is_train=True):
    if not is_train:
        transfroms = T.Compose([
            T.Resize(cfg.size_test, interpolation=3),
            T.ToTensor(),
            nn.InstanceNorm2d(3, affine=False)
        ])
        return transfroms

    transfroms = T.Compose([
        T.Resize(cfg.size_train, interpolation=3),
        T.RandomGrayscale(p=cfg.gray_scale),
        T.Pad(cfg.pad),
        T.RandomCrop(cfg.size_train),
        T.RandomHorizontalFlip(p=cfg.random_horizontal_flip),
        T.RandomApply([T.ColorJitter(0.15, 0.1, 0.05, 0.05)], p=cfg.color_jitter),
        T.AugMix(prob=cfg.aug_mix),
        T.ToTensor(),
        T.RandomErasing(cfg.random_erasing),
        nn.InstanceNorm2d(3)
    ])
    return transfroms
