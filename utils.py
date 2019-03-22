#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# @author = yuci
# Date: 19-3-18 上午11:11
"""第二步 工具集合
define some utils in this file.
Code imported from https://github.com/Cysu/open-reid/.
"""
import torch
import os
import os.path as osp
import sys
import errno
import json
import shutil


# 0.to_numpy/to_torch 转换数据类型
def to_numpy(tensor):
	if torch.is_tensor(tensor):
		return tensor.cpu().numpy()
	elif type(tensor).__module__ != 'numpy':
		raise ValueError("Cannot convert {} to numpy array".format(type(tensor)))
	return tensor


def to_torch(ndarray):
	if type(ndarray).__module__ == 'numpy':
		return torch.from_numpy(ndarray)
	elif not torch.is_tensor(ndarray):
		raise ValueError("Cannot convert {} to torch tensor".format(type(ndarray)))
	return ndarray


# 1.makefile/dir如果对应路径中的文件不存在，则新建这个文件
def mkdir_if_missing(dir_path):
	try:
		os.makedirs(dir_path)
	except OSError as e:
		if e.errno != errno.EEXIST:
			raise


# 2.Computes and stores the average and current value  计算和存储平均值、当前值，输出损失时用到
class AverageMeter(object):
	def __init__(self):
		self.val = 0
		self.avg = 0
		self.sum = 0
		self.count = 0

	def reset(self):
		self.val = 0
		self.avg = 0
		self.sum = 0
		self.count = 0

	def update(self, val, n=1):
		self.val = val
		self.sum += val * n
		self.count += n
		self.avg = self.sum / self.count


# 3. logger 日志管理 Write console output to external text file.
class Logger(object):
	def __init__(self, fpath=None):
		self.console = sys.stdout
		self.file = None
		if fpath is not None:
			mkdir_if_missing(os.path.dirname(fpath))
			self.file = open(fpath, 'w')

	def __del__(self):
		self.close()

	def __enter__(self):
		pass

	def __exit__(self):
		self.close()

	def write(self, msg):
		self.console.write(msg)
		if self.file is not None:
			self.file.write(msg)

	def flush(self):
		self.console.flush()
		if self.file is not None:
			self.file.flush()
			os.fsync(self.file.fileno())

	def close(self):
		self.console.close()
		if self.file is not None:
			self.file.close()


# 4.serialization
def read_json(fpath):
	with open(fpath, 'r') as f:
		obj = json.load(f)  # 解码：把Json格式字符串解码转换成Python对象
	return obj


def write_json(obj, fpath):
	mkdir_if_missing(osp.dirname(fpath))  # 对应的文件如果不存在，则新建它
	with open(fpath, 'w') as f:
		json.dump(obj, f, indent=4, separators=(',', ':'))  # 这表示dictionary内keys之间用“,”隔开，而KEY和value之间用“：”隔开
	# 使用json.dump()将数据obj写入文件f，会换行且按照indent的数量显示前面的空白，


def save_checkpoint(state, is_best, fpath='checkpoint.pth.tar'):  # 保存checkpoint文件
	mkdir_if_missing(osp.dirname(fpath))
	torch.save(state, fpath)
	if is_best:
		shutil.copy(fpath, osp.join(osp.dirname(fpath), 'model_best.pth.tar'))


def load_checkpoint(fpath):  # 加载checkpoint文件
	if osp.isfile(fpath):
		checkpoint = torch.load(fpath)
		print("=> Loaded checkpoint '{}'".format(fpath))
		return checkpoint
	else:
		raise ValueError("=> No checkpoint found at '{}'".format(fpath))
