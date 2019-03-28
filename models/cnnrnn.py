#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# @author = yuci
# Date: 19-3-21 下午7:29
"""CNN-RNN 网络架构  三层CNN
1.第一层网络：16个卷积核，尺寸为5*5，步长为2；2*2最大池化；tanh激活函数
2.第二层网络：64个卷积核，尺寸为5*5，步长为2；2*2最大池化；tanh激活函数
3.第三层网络：64个卷积核，尺寸为5*5，步长为2；2*2最大池化；tanh激活函数
4.0.5的dropout
5.128个元素的FC全连接层

"""
import torch
import torch.nn as nn
import numpy as np
from torch.nn import functional as F


class Net(nn.Module):
	def __init__(self, nFilter1, nFilter2, nFilter3, num_person_train, dropout=0.5, num_features=0, seq_len=0, batch=0):
		super(Net, self).__init__()
		self.batch = batch
		self.seq_len = seq_len
		self.num_person_train = num_person_train
		self.dropout = dropout  # 随机失活的概率，0-1
		self.num_features = num_features  # 输出的特征维度 128
		self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

		self.nFilters = [nFilter1, nFilter2, nFilter3]  # 初始化每一层的卷积核个数
		self.filter_size = [5, 5, 5]  # 卷积核尺寸

		self.poolsize = [2, 2, 2]  # 最大池化的尺寸
		self.stepsize = [2, 2, 2]  # 池化步长
		self.padDim = 4  # 零填充
		self.input_channel = 5  # 3img + 2optical flow

		# 构建卷积层，nn。Conv2d(输入通道，卷积核的个数，卷积核尺寸，步长，零填充)
		self.conv1 = nn.Conv2d(self.input_channel, self.nFilters[0], self.filter_size[0], stride=1, padding=self.padDim)
		self.conv2 = nn.Conv2d(self.nFilters[0], self.nFilters[1], self.filter_size[1], stride=1, padding=self.padDim)
		self.conv3 = nn.Conv2d(self.nFilters[1], self.nFilters[2], self.filter_size[2], stride=1, padding=self.padDim)

		# 构建最大池化层
		self.pooling1 = nn.MaxPool2d(self.poolsize[0], self.stepsize[0])
		self.pooling2 = nn.MaxPool2d(self.poolsize[1], self.stepsize[1])
		self.pooling3 = nn.MaxPool2d(self.poolsize[2], self.stepsize[2])

		# tanh激活函数
		self.tanh = nn.Tanh()

		# FC层
		n_fully_connected = 21280  # 根据图片尺寸修改

		self.seq2 = nn.Sequential(
			nn.Dropout(self.dropout),
			nn.Linear(n_fully_connected, self.num_features)
		)

		# rnn层
		self.rnn = nn.RNN(self.num_features, self.num_features)
		self.hid_weight = nn.Parameter(
			nn.init.xavier_uniform_(
				torch.Tensor(1, self.seq_len * self.batch, self.num_features).to(self.device), gain=np.sqrt(2.0)
			), requires_grad=True,
		)

		# final full connectlayer
		self.final_FC = nn.Linear(self.num_features, self.num_person_train)

	def build_net(self, input1, input2):
		seq1 = nn.Sequential(
			self.conv1, self.tanh, self.pooling1,
			self.conv2, self.tanh, self.pooling2,
			self.conv3, self.tanh, self.pooling3,
		)
		b = input1.size(0)  # batch的大小
		n = input1.size(1)  # 1个batch中图片的数目
		input1 = input1.view(b*n, input1.size(2), input1.size(3), input1.size(4))  #
		input2 = input2.view(b*n, input2.size(2), input2.size(3), input2.size(4))
		inp1_seq1_out = seq1(input1).view(input1.size(0), -1)  # torch.Size([16, 32, 35, 19])
		inp2_seq1_out = seq1(input2).view(input2.size(0), -1)  # 经过卷积层后的输出   torch.Size([16, 32, 35, 19])
		inp1_seq2_out = self.seq2(inp1_seq1_out).unsqueeze_(0)
		inp2_seq2_out = self.seq2(inp2_seq1_out).unsqueeze_(0)  # 经过fc层的输出

		inp1_rnn_out, hn1 = self.rnn(inp1_seq2_out, self.hid_weight)
		inp2_rnn_out, hn2 = self.rnn(inp2_seq2_out, self.hid_weight)   # todo:should debug here
		inp1_rnn_out = inp1_rnn_out.view(b, n, -1)  # torch.Size([8, 16, 128])
		inp2_rnn_out = inp2_rnn_out.view(b, n, -1)
		inp1_rnn_out = inp1_rnn_out.permute(0, 2, 1)
		inp2_rnn_out = inp2_rnn_out.permute(0, 2, 1)  # 8,128,16

		# 平均池化/最大池化
		feature_p = F.max_pool1d(inp1_rnn_out, inp1_rnn_out.size(2))  # 序列特征 8, 128, 1
		feature_g = F.max_pool1d(inp2_rnn_out, inp2_rnn_out.size(2))

		feature_p = feature_p.view(b, self.num_features)  # 8,128
		feature_g = feature_g.view(b, self.num_features)

		# 分类
		identity_p = self.final_FC(feature_p)  # 身份特征 torch.Size([8, 89])
		identity_g = self.final_FC(feature_g)
		return feature_p, feature_g, identity_p, identity_g

	def forward(self, input1, input2):
		feature_p, feature_g, identity_p, identity_g = self.build_net(input1, input2)
		return feature_p, feature_g, identity_p, identity_g


class Criterion(nn.Module):
	def __init__(self, hinge_margin=2):
		super(Criterion, self).__init__()
		self.hinge_margin = hinge_margin
		self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

	def forward(self, feature_p, feature_g, identity_p, identity_g, target):
		dist = nn.PairwiseDistance(p=2)
		pair_dist = dist(feature_p, feature_g)  # 欧几里得距离

		# 1.折页损失
		hing = nn.HingeEmbeddingLoss(margin=self.hinge_margin, reduce=False)
		label0 = target[0].to(self.device)
		hing_loss = hing(pair_dist, label0)

		# 2.交叉熵损失
		nll = nn.CrossEntropyLoss()
		label1 = target[1].to(self.device)
		label2 = target[2].to(self.device)
		loss_p = nll(identity_p, label1)
		loss_g = nll(identity_g, label2)

		# 3.损失求和
		total_loss = hing_loss + loss_p + loss_g
		mean_loss = torch.mean(total_loss)

		return mean_loss



