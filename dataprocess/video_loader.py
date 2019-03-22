#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# @author = yuci
# Date: 19-3-20 下午8:38
""" 第四步"""
from PIL import Image
import numpy as np
import random

import torch
from torch.utils.data import Dataset
from utils import to_torch


class VideoDataset(Dataset):
	"""形成一个批次的数据 (批次大小，序列长度，channel， height，width)"""
	sample_methods = ['random', 'dense']

	def __init__(self, dataset, seq_len=16, sample='random', transform=None):
		self.dataset = dataset  # 数据集
		self.seq_len = seq_len  # 批次数据中的序列长度
		self.sample = sample  # 采样方法，随机采样16个就结束，还是密集采样所有
		self.transform = transform  # 对数据集中的图片进行数据增强

	def __len__(self):  # 获得数据集的长度
		return len(self.dataset)

	def __getitem__(self, item):
		img_paths, pid, camid = self.dataset[item]  # 从数据集中获得某个id相应的信息
		num = len(img_paths)  # 获得这个id下，有多少张图片

		if self.sample == 'random':
			"""
			Randomly sample seq_len consecutive frames from num frames,
			if num is smaller than seq_len, then replicate items.
			This sampling strategy is used in training phase.
			"""
			frame_indices = list(range(num))  # 将这些图片建立索引
			rand_end = max(0, len(frame_indices) - self.seq_len - 1)  # 随机选择的索引起点最大值，
			begin_index = random.randint(0, rand_end)  # 序列起点索引
			end_index = min(begin_index + self.seq_len, len(frame_indices))  # 序列终点索引
			indices = frame_indices[begin_index:end_index]  # 根据随机选择的索引，确定序列的索引

			for index in indices:  # 遍历序列索引，
				if len(indices) >= self.seq_len:
					break
				indices.append(index)  # 当序列长度小于采样的长度时，复制序列，直到等于seq_len

			indices = np.array(indices)  # 列表转换成数组
			imgseq = []  # 存放图像序列

			for index in indices:
				index = int(index)
				img_paths = img_paths[index]  # 获得对应索引index下的图像绝对路径
				img = Image.open(img_paths).convert('RGB')  # 将图片转换成RGB，　3*256*128

				if self.transform is not None:
					img = self.transform(img)  # 进行数据增强

				img = img.unsqueeze(0)  # 在第０维增加一个维度  1*3*256*128
				imgseq.append(img)
			imgseq = to_torch(imgseq)
			imgseq = torch.cat(imgseq, dim=0)  # seq_len*3*256*128

			return imgseq, pid, camid
		elif self.sample == 'dense':
			"""
			Sample all frames in a video into a list of clips,
			each clip contains seq_len frames, batch_size needs to be set to 1.
			This sampling strategy is used in test phase.
			"""
			cur_index = 0  # 密集采样，起始索引为0
			frame_indices = list(range(num))  # 图像帧索引列表
			indices_list = []

			while num - cur_index > self.seq_len:  # 当序列总长度-当前索引 > 采样长度，则更新当前索引，一直遍历这个序列
				indices_list.append(frame_indices[cur_index:cur_index+self.seq_len])
				cur_index += self.seq_len  # 更新当前索引
			last_seq = frame_indices[cur_index:]  # 最后一个索引不满足采样长度,补足最后一个
			for index in last_seq:
				if len(last_seq) > self.seq_len:
					break
				last_seq.append(index)
			indices_list.append(last_seq)  # 加上最后一个采样长度

			imgs_list = []
			for indices in indices_list:  # 遍历每一个采样序列长度
				imgseq = []  # 用于存放一个采样序列长度的图片
				for index in indices:  # 遍历每个采样序列中的每一张图片
					index = int(index)
					img_path = img_paths[index]
					img = Image.open(img_path).convert('RGB')
					if self.transform is not None:
						img = self.transform(img)
					img = img.unsqueeze(0)  # [1, 3, 224, 112]
					imgseq.append(img)
				imgseq = to_torch(imgseq)
				imgseq = torch.cat(imgseq, dim=0)  # [seq_len, 3, 224, 112]获得一个采样序列的图像
				imgs_list.append(imgseq)
			imgs_list = tuple(imgs_list)
			imgs_array = torch.stack(imgs_list)  # 获得整个行人序列的密集采样序列

			return imgs_array, pid, camid
		else:
			raise KeyError("unknown sample method: {}.".format(self.sample))
