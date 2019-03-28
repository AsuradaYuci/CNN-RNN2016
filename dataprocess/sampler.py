#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# @author = yuci
# Date: 19-3-25 上午9:42
import numpy as np
import torch

from torch.utils.data.sampler import (Sampler, SequentialSampler)
from collections import defaultdict


# 构成正序列对时，从另一个摄像头中选择对应的行人数据
def No_index(a, b):
	assert isinstance(a, list)
	return [i for i, j in enumerate(a) if j != b]


class RandomPairSampler(Sampler):

	def __init__(self, data_source):
		self.data_source = data_source  # data_source的结构是一个元组（img_path, pid, cam_id）
		self.index_pid = defaultdict(int)  # 索引对应的pid ：index ---pid
		self.pid_cam = defaultdict(list)  # pid对应的cam,是个列表
		self.pid_index = defaultdict(list)  # pid 对应的索引
		self.num_samples = len(self.data_source)  # 数据集的长度即为采样的数目  178

		for index, (_, pid, cam) in enumerate(self.data_source):
			self.index_pid[index] = pid
			self.pid_cam[pid].append(cam)
			self.pid_index[pid].append(index)

	def __len__(self):
		return self.num_samples * 2  # 采样后的数据是原数据集长度的两倍

	def __iter__(self):  # 返回正负序列对
		indices = torch.randperm(self.num_samples)
		ret = []  # 返回（seqA, seqB, target）
		for i in range(2*self.num_samples):

			if i % 2 == 0:  # positive pair
				j = i // 2
				j = int(indices[j])  # 确定序列对的第一个序列j
				_, j_pid, j_cam = self.data_source[j]
				pid_j = self.index_pid[j]
				cams = self.pid_cam[pid_j]
				index = self.pid_index[pid_j]
				select_cams = No_index(cams, j_cam)  # 从另一个cam中选择第二个序列
				try:
					select_camind = np.random.choice(select_cams)
				except ValueError:
					print(cams)
					print(pid_j)
				select_ind = index[select_camind]  # 选择第二个序列
				target = [1, pid_j, pid_j]  # 标签信息
				ret.append((j, select_ind, target))
			else:  # negative pair
				p_rand_id = torch.randperm(self.num_samples)
				a = int(p_rand_id[0])  # 随机选择
				pid_a = self.index_pid[a]

				b = int(p_rand_id[1])
				_, b_pid, b_cam = self.data_source[b]
				pid_b = self.index_pid[b]
				cams = self.pid_cam[pid_b]
				index = self.pid_index[pid_b]
				select_cams = No_index(cams, b_cam)
				try:
					select_camind = np.random.choice(select_cams)
				except ValueError:
					print(cams)
					print(pid_b)
				select_ind = index[select_camind]

				target = [-1, pid_a, pid_b]
				ret.append((a, select_ind, target))
		return iter(ret)
