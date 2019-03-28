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
import os.path as osp
import cv2 as cv


class VideoDataset(Dataset):
	"""形成一个批次的数据 (批次大小，序列长度，channel， height，width)"""
	sample_methods = ['random', 'dense']

	def __init__(self, dataset, seq_len=16, sample='random', transform=None):
		self.dataset = dataset  # 数据集
		self.seq_len = seq_len  # 批次数据中的序列长度
		self.sample = sample  # 采样方法，随机采样16个就结束，还是密集采样所有
		self.transform = transform  # 对数据集中的图片进行数据增强
		self.useOpticalFlow = 1

	def __len__(self):  # 获得数据集的长度
		return len(self.dataset)

	def getOfPath(self, rgb_path):  # (一次1张图片)将图像转换为YUV，并且加入光流信息
		root_of = '/home/ying/Desktop/video_reid_mars/data/prid2011sequence/raw/prid2011flow/prid2011flow'
		fname_list = rgb_path.split('/')
		of_path = osp.join(root_of, fname_list[-3], fname_list[-2], fname_list[-1])  # 光流路径

		return of_path

	def __getitem__(self, item):  # item = tuple(103, 102, [1, 51, 51])
		if self.sample == 'random':
			"""
			Randomly sample seq_len consecutive frames from num frames,
			if num is smaller than seq_len, then replicate items.
			This sampling strategy is used in training phase.
			"""
			item0, item1, target = item
			img0_paths, pid0, camid0 = self.dataset[item0]

			img1_paths, pid1, camid1 = self.dataset[item1]  # 从数据集中获得某个id相应的信息
			num0 = len(img0_paths)  # 获得这个id下，有多少张图片  73
			num1 = len(img1_paths)  # 119
			# seq0
			frame_indices0 = list(range(num0))  # 将这些图片建立索引 [0, 1,...,72]
			rand_end0 = max(0, len(frame_indices0) - self.seq_len - 1)  # 随机选择的索引起点最大值，56
			begin_index0 = random.randint(0, rand_end0)  # 序列起点索引 0
			end_index0 = min(begin_index0 + self.seq_len, len(frame_indices0))  # 序列终点索引  16
			indices0 = frame_indices0[begin_index0:end_index0]  # 根据随机选择的索引，确定序列的索引

			for index0 in indices0:  # 遍历序列索引，
				if len(indices0) >= self.seq_len:
					break
				indices0.append(index0)  # 当序列长度小于采样的长度时，复制序列，直到等于seq_len

			indices0 = np.array(indices0)  # 列表转换成数组
			imgseq0 = []  # 存放图像序列
			flowseq0 = []

			for index0 in indices0:  # [0, 1,...,15]
				index = int(index0)  # 0
				img_paths0 = img0_paths[index]  # 获得对应索引index下的图像绝对路径
				of_paths0 = self.getOfPath(img_paths0)  # 获得对应的光流图片
				imgrgb0 = Image.open(img_paths0).convert('RGB')
				ofrgb0 = Image.open(of_paths0).convert('RGB')
				imgseq0.append(imgrgb0)
				flowseq0.append(ofrgb0)
			seq0 = [imgseq0, flowseq0]  # [['.png','.png']]
			if self.transform is not None:  # 以序列为单位进行数据增强，同时在transform中将图像转换为YUV格式
				seq0 = self.transform(seq0)  # 进行数据增强,在transform中将光流数据加上
			img_tensor0 = seq0  # torch.Size([16, 5, 128, 64])

			# seq1  todo: 参照上面的修改
			frame_indices1 = list(range(num1))
			rand_end1 = max(0, len(frame_indices1) - self.seq_len - 1)
			begin_index1 = random.randint(0, rand_end1)
			end_index1 = min(begin_index1 + self.seq_len, len(frame_indices1))
			indices1 = frame_indices1[begin_index1:end_index1]

			for index1 in indices1:
				if len(indices1) >= self.seq_len:
					break
				indices1.append(index1)

			indices1 = np.array(indices1)
			imgseq1 = []
			flowseq1 = []

			for index1 in indices1:
				index = int(index1)
				img_paths1 = img1_paths[index]
				of_paths1 = self.getOfPath(img_paths1)
				imgrgb1 = Image.open(img_paths1)
				ofrgb1 = Image.open(of_paths1)
				imgseq1.append(imgrgb1)
				flowseq1.append(ofrgb1)
			seq1 = [imgseq1, flowseq1]
			if self.transform is not None:
				seq1 = self.transform(seq1)
			img_tensor1 = seq1

			return img_tensor0, img_tensor1, target
		elif self.sample == 'dense':
			"""
			Sample all frames in a video into a list of clips,
			each clip contains seq_len frames, batch_size needs to be set to 1.
			This sampling strategy is used in test phase.
			"""
			img_paths, pid, camid = self.dataset[item]
			num = len(img_paths)  # 27
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
