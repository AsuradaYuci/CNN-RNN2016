#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# @author = yuci
# Date: 19-3-18 上午11:03
from .cnnrnn import Net


__factory = {
	'cnn-rnn': Net,
}


def names():
	return sorted(__factory.keys())


def creat(name, *args, **kwargs):
	if name not in __factory:
		raise KeyError("unknown model:", name)
	return __factory[name](*args, **kwargs)
