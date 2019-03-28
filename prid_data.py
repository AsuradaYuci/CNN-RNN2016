#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# @author = yuci
# Date: 19-3-18 上午11:04
"""第一步 main文件"""

import torch
import numpy as np
import sys
import argparse
import os.path as osp
import torch.backends.cudnn as cudnn
# import torchvision.transforms as T
from torch.utils.data import DataLoader
import torch.optim as optim

from utils import Logger, load_checkpoint, save_checkpoint
from dataset import get_sequence
from dataprocess.sampler import RandomPairSampler
from dataprocess.video_loader import VideoDataset
from dataprocess import seqtransform as T
import models
from models.cnnrnn import Criterion
from eval import Evaluator
from train import SEQTrainer


# get dataset 数据集准备
def getdata(dataset_name, split_id, batch_size, seq_len, seq_srd, workers):
	# todo:将光流信息加上，路径问题，数据增强
	# root_rgb = '/home/ying/Desktop/video_reid_mars/data/prid2011sequence/raw/prid_2011/prid_2011/multi_shot'
	# root_of = '/home/ying/Desktop/video_reid_mars/data/prid2011sequence/raw/prid2011flow'
	dataset = get_sequence(dataset_name, split_id)
	train_set = dataset.train
	num_classes = dataset.num_train_pids  # 89
	# normalizer = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
	# 采用的数据增强：随机裁剪，水平翻转
	transform_train = T.Compose([
		T.RectScale(256, 128),
		T.RandomSizedEarser(),
		T.RandomHorizontalFlip(),
		T.ToYUV(),
	])
	transform_query = T.Compose([
		T.RectScale(256, 128),
		T.ToYUV(),
	])
	transform_gallery = T.Compose([
		T.RectScale(256, 128),
		T.ToYUV(),
	])
	# 对数据集进行处理，封装进Dataloader   ==> 类的初始化
	train_processor = VideoDataset(dataset.train, seq_len=seq_len, sample='random', transform=transform_train)
	query_processor = VideoDataset(dataset.query, seq_len=seq_len, sample='dense', transform=transform_query)
	gallery_processor = VideoDataset(dataset.gallery, seq_len=seq_len, sample='dense', transform=transform_gallery)

	train_loader = DataLoader(train_processor, batch_size=batch_size, num_workers=workers, sampler=RandomPairSampler(train_set), pin_memory=True, drop_last=True)
	query_loader = DataLoader(query_processor, batch_size=1, num_workers=workers, shuffle=False, pin_memory=True, drop_last=False)
	gallery_loader = DataLoader(gallery_processor, batch_size=1, num_workers=workers, shuffle=False, pin_memory=True, drop_last=False)

	return dataset, num_classes, train_loader, query_loader, gallery_loader


def main(args):
	# 1.初始化设置
	np.random.seed(args.seed)
	torch.manual_seed(args.seed)
	torch.cuda.manual_seed_all(args.seed)  # 为GPU设置随机数种子
	cudnn.benchmark = True  # 在程序刚开始加这条语句可以提升一点训练速度,没什么额外开销
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

	# 2.日志文件 log
	if args.evaluate == 1:
		sys.stdout = Logger(osp.join(args.logs_dir, 'log_test.txt'))
	else:
		sys.stdout = Logger(osp.join(args.logs_dir, 'log_train.txt'))

	# 3.数据集 todo:重建这里的数据集
	dataset, numclasses, train_loader, query_loader, gallery_loader = getdata(args.dataset, args.split, args.batch_size, args.seq_len, args.seq_srd, args.workers)

	# 4.建立网络
	cnn_rnn_model = models.creat(args.a1, 16, 32, 32, numclasses, num_features=args.features, seq_len=args.seq_len, batch=args.batch_size).to(device)
	criterion = Criterion(args.hingeMargin).to(device)
	optimizer = optim.SGD(cnn_rnn_model.parameters(), lr=args.lr1, momentum=args.momentum, weight_decay=args.weight_decay)

	# 5.trainer实例化
	trainer = SEQTrainer(cnn_rnn_model, criterion)

	# 6.evaluate类的实例化
	evaluator = Evaluator(cnn_rnn_model)
	best_top1 = 0

	# 6.进入训练/测试模式
	if args.evaluate == 1:
		checkpoint = load_checkpoint(osp.join(args.logs_dir, 'cnn_rnn_best.pth.tar'))
		cnn_rnn_model.load_state_dict(checkpoint['state_dict'])
		rank1 = evaluator.evaluate(query_loader, gallery_loader, dataset.queryinfo, dataset.galleryinfo)
	else:
		cnn_rnn_model.train()
		for epoch in range(args.start_epoch, args.epochs):
			trainer.train(epoch, train_loader, optimizer)

			# 每隔几个epoch测试一次
			if (epoch + 1) % 400 == 0 or (epoch + 1) == args.epochs:

				top1 = evaluator.evaluate(query_loader, gallery_loader, dataset.queryinfo, dataset.galleryinfo)
				is_best = top1 > best_top1
				if is_best:
					best_top1 = top1

				save_checkpoint({
					'state_dict': cnn_rnn_model.state_dict(),
					'epoch': epoch + 1,
					'best_top1': best_top1,
				}, is_best, fpath=osp.join(args.logs_dir, 'cnn_checkpoint.pth.tar'))


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='hh')
	parser.add_argument('--seed', type=int, default=1)
	# DATA
	parser.add_argument('--dataset', type=str, default='prid2011', choices=['ilds', 'prid2011', 'mars'])
	parser.add_argument('--batch-size', type=int, default=8, help='depend on your device')
	parser.add_argument('--workers', type=int, default=4)
	parser.add_argument('--seq_len', type=int, default=16, help='the number of images in a sequence')
	parser.add_argument('--seq_srd', type=int, default=16, help='采样间隔步长')
	parser.add_argument('--split', type=int, default=0, help='total 10')
	# Model
	parser.add_argument('--a1', type=str, default='cnn-rnn')
	parser.add_argument('--nConvFilters', type=int, default=32)
	parser.add_argument('--features', type=int, default=128, help='features dimension')
	parser.add_argument('--hingeMargin', type=int, default=2)
	# parser.add_argument('--dropout', type=float, default=0.0)
	# Optimizer
	parser.add_argument('--lr1', type=float, default=0.001)
	parser.add_argument('--lrstep', type=int, default=20)
	parser.add_argument('--momentum', type=float, default=0.9)
	parser.add_argument('--weight-decay', type=float, default=5e-4)
	# Train
	parser.add_argument('--start-epoch', type=int, default=0)
	parser.add_argument('--epochs', type=int, default=400)
	parser.add_argument('--evaluate', type=int, default=0, help='0 => train; 1 =>test')
	# Path
	working_dir = osp.dirname(osp.abspath(__file__))
	parser.add_argument('--dataset-dir', type=str, metavar='PATH', default=osp.join(working_dir, '../video_reid_mars/data'))
	parser.add_argument('--logs-dir', type=str, metavar='PATH', default=osp.join(working_dir, 'log/yuci'))
	args = parser.parse_args()
	# main func
	main(args)
