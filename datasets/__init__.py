# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

from reidutils.registry import Registry

DATASET_REGISTRY = Registry("DATASET")
DATASET_REGISTRY.__doc__ = """
Registry for datasets
It must returns an instance of :class:`Backbone`.
"""

# Person re-id datasets
from .cuhk03 import cuhk03
from .market1501 import Market
from .msmt17 import MSMT17
from .iLIDS import iLIDS
from .cuhk_sysu import cuhk_sysu
from .cuhk02 import CUHK02
from .grid import GRID
from .prid import PRID
from .viper import VIPeR


__all__ = [k for k in globals().keys() if "builtin" not in k and not k.startswith("_")]
