#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# @author = yuci
# Date: 19-3-18 上午11:01


from .prid2011 import PRID


def get_sequence(name, *args, **kwargs):
    __factory = {
        'prid2011sequence': PRID,
    }

    if name not in __factory:
        raise KeyError("Unknown dataset", name)
    return __factory[name](*args, **kwargs)
