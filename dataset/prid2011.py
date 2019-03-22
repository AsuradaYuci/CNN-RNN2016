#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# @author = yuci
# Date: 19-3-18 下午9:28
import glob
import os.path as osp
import numpy as np
from utils import read_json

"""第三步 数据集形成"""


# 用于构建数据集信息
class infostruct(object):
	pass


# 1.datasequence,数据集的总体情况
class PRID(object):
	"""
	code mainly from https://github.com/KaiyangZhou/deep-person-reid
	"""
	root = '/home/ying/Desktop/video_reid_mars/data/prid2011sequence/raw/prid_2011'
	split_path = osp.join(root, 'splits_prid2011.json')
	cam_a_path = osp.join(root, 'prid_2011', 'multi_shot', 'cam_a')
	cam_b_path = osp.join(root, 'prid_2011', 'multi_shot', 'cam_b')

	def __init__(self, split_id=0, min_seq_len=0):
		self._check_before_run()  # 一、检查数据集是否存在

		splits = read_json(self.split_path)  # 二、读取split文件，包含数据集分割信息train/test
		if split_id >= len(splits):  # split是个元组？查询的split_id 不能超过整个分割的长度
			raise ValueError("split id exceeds range, received {}, but expected between 0 and {}".format(split_id, len(self.split_path)))
		split = splits[split_id]  # 根据split_id从splits文件中选出对应的分割split
		train_split, test_split = split['train'], split['test']
		print("# train identites: {}, # test identites: {}".format(len(train_split), len(test_split)))

		# 三、根据split信息，处理原始数据集。返回数据集，训练集中tracklet的数量，行人id的数量，每个tracklet中图片的数量
		train, num_train_tracklets, num_train_pids, num_imgs_train = \
			self._process_data(train_split, cam1=True, cam2=True)
		query, num_query_tracklets, num_query_pids, num_imgs_query = \
			self._process_data(test_split, cam1=True, cam2=False)
		gallery, num_gallery_tracklets, num_gallery_pids, num_imgs_gallery = \
			self._process_data(test_split, cam1=False, cam2=True)

		# 四、统计下每个视频序列的长度信息
		num_imgs_per_tracklet = num_imgs_train + num_imgs_query + num_imgs_gallery  # 列表
		min_num = np.min(num_imgs_per_tracklet)
		max_num = np.max(num_imgs_per_tracklet)
		avg_num = np.mean(num_imgs_per_tracklet)

		# 五、统计行人id信息
		num_total_pids = num_train_pids + num_query_pids + num_gallery_pids  # 一个数
		num_total_tracklets = num_train_tracklets + num_query_tracklets + num_gallery_tracklets

		# 六、封装数据集？
		self.train = train
		self.query = query
		self.gallery = gallery
		self.num_train_pids = num_train_pids
		self.num_query_pids = num_query_pids
		self.num_gallery_pids = num_gallery_pids

		# 七、打印数据集的一些基本信息
		print("=> PRID-2011 loaded")
		print("Dataset statistics:")
		print("  ------------------------------")
		print("  subset   | # ids | # tracklets")
		print("  ------------------------------")
		print("  train    | {:5d} | {:8d}".format(num_train_pids, num_train_tracklets))
		print("  query    | {:5d} | {:8d}".format(num_query_pids, num_query_tracklets))
		print("  gallery  | {:5d} | {:8d}".format(num_gallery_pids, num_gallery_tracklets))
		print("  ------------------------------")
		print("  total    | {:5d} | {:8d}".format(num_total_pids, num_total_tracklets))
		print("  number of images per tracklet: {} ~ {}, average {:.1f}".format(min_num, max_num, avg_num))
		print("  ------------------------------")

	# check if all files are available
	def _check_before_run(self):
		if not osp.exists(self.root):
			raise RuntimeError("'{}' is not available".format(self.root))

	# 根据split文件，处理原始数据集，形成对应的训练集，测试集
	def _process_data(self, dirnames, cam1=True, cam2=True):
		tracklets = []  # 列表，存放每个id的视频序列, 行人id，cam id
		num_imgs_per_tracklet = []  # 统计每个tracklet中的图片数目
		dirname2pid = {dirname: i for i, dirname in enumerate(dirnames)}  # 根据数据集特点，将文件名字转换为连续的id

		for dirname in dirnames:
			if cam1:  # cam_a 摄像头中数据
				person_dir = osp.join(self.cam_a_path, dirname)  # 获得对应行人id的文件夹路径
				img_names = glob.glob(osp.join(person_dir, '*.png'))  # 使用glob函数，获取文件夹中所有图片的路径
				assert len(img_names) > 0
				img_names = tuple(img_names)  # 将所有图片路径存在一个元组中
				pid = dirname2pid[dirname]  # 根据数据集特点，将文件夹名字转换为行人id
				tracklets.append((img_names, pid, 0))
				num_imgs_per_tracklet.append(len(img_names))
			if cam2:
				person_dir = osp.join(self.cam_b_path, dirname)  # 获得对应行人id的文件夹路径
				img_names = glob.glob(osp.join(person_dir, '*.png'))  # 使用glob函数，获取文件夹中所有图片的路径
				assert len(img_names) > 0
				img_names = tuple(img_names)  # 将所有图片路径存在一个元组中
				pid = dirname2pid[dirname]  # 根据数据集特点，将文件夹名字转换为行人id
				tracklets.append((img_names, pid, 1))
				num_imgs_per_tracklet.append(len(img_names))

		num_tracklets = len(tracklets)  # 最后统计下视频序列的个数
		num_pid = len(dirnames)  # 统计行人的id数

		return tracklets, num_tracklets, num_pid, num_imgs_per_tracklet

