import os

import torch
import numpy as np
import cv2 as cv
import fnmatch


class Prepare():
    def __init__(self, args):
        self.args = args
        pass

    # get person directory 从第一个摄像头的文件夹中获得385的行人id对应的视频序列
    def getPersonDirsList(self, rgb_dirDir):
        if self.args.dataset == 1 or self.args.dataset == 0:
            firstCameraDirName = 'cam1'
        else:
            firstCameraDirName = 'cam_1'
        seqDir = rgb_dirDir + '/' + firstCameraDirName  # '/home/ying/Desktop/video_reid_mars/data/prid2011sequence/raw/prid_2011/prid_2011/multi_shot//cam_a'
        personDir = os.listdir(seqDir)  # os.listdir() 方法用于返回指定的文件夹包含的文件或文件夹的名字的列表  385个元素的列表，乱序
        if len(personDir) == 0:
            return False
        for i, dir in enumerate(personDir):
            if len(dir) <= 2:
                del personDir[i]
        personDir.sort()  # 列表中的元素 从小到大排序
        return personDir

    # get name list of images  获得对应person_001文件夹中的所有图片名称
    def getSequenceImages(self, rgb_dir, file_suffix):
        img_list = os.listdir(rgb_dir)
        img = fnmatch.filter(img_list, '*.' + file_suffix)
        img.sort()
        return img  # <class 'list'>: ['0001.png', '0002.png', ..., '0140.png', '0141.png', '0142.png']

    # load images into tensor
    def loadSequenceImages(self, rgb_dir, of_dir, img_list):
        nImgs = len(img_list)  # 142
        # print(nImgs, rgb_dir, of_dir)
        imagePixelData = torch.zeros((nImgs, 5, 64, 48), dtype=torch.float32, device=torch.device('cuda', 0))
        # torch.Size([142, 5, 64, 48])
        for i, file in enumerate(img_list):  # i=0,file='0001.png'
            filename = '/'.join([rgb_dir, file])  # '/home/ying/Desktop/Spatial-Temporal-Pooling-Networks-ReID-master/data/prid_2011/multi_shot//cam_1/person_0001/0001.png'
            filename_of = '/'.join([of_dir, file])  # '/home/ying/Desktop/Spatial-Temporal-Pooling-Networks-ReID-master/data/prid2011flow/cam_1/person_0001/0001.png'
            img = cv.imread(filename)  # <class 'tuple'>: (128, 64, 3)  (h,w,c)
            img = cv.resize(img, (48, 64))  # <class 'tuple'>: (64, 48, 3)
            img = img.astype(np.float32)

            imgof = cv.imread(filename_of)  # <class 'tuple'>: (128, 64, 3)
            imgof = cv.resize(imgof, (48, 64))  # <class 'tuple'>: (64, 48, 3)
            imgof = imgof.astype(np.float32)

            # change image to YUV channels
            img = cv.cvtColor(img, cv.COLOR_BGR2YUV)
            img_tensor = torch.from_numpy(img).type(torch.float32)  # torch.Size([64, 48, 3])
            imgof_tensor = torch.from_numpy(imgof).type(torch.float32)  # torch.Size([64, 48, 3])
            for c in range(3):
                v = torch.sqrt(torch.var(img_tensor[:, :, c]))  # v = tensor(28.5955)
                m = torch.mean(img_tensor[:, :, c])  # m = tensor(126.2291)
                img_tensor[:, :, c] = img_tensor[:, :, c] - m
                img_tensor[:, :, c] = img_tensor[:, :, c] / torch.sqrt(v)
                imagePixelData[i, c] = img_tensor[:, :, c]

            for c in range(2):
                v = torch.sqrt(torch.var(imgof_tensor[:, :, c]))
                m = torch.mean(imgof_tensor[:, :, c])
                imgof_tensor[:, :, c] = imgof_tensor[:, :, c] - m
                imgof_tensor[:, :, c] = imgof_tensor[:, :, c] / torch.sqrt(v)
                imagePixelData[i, c + 3] = imgof_tensor[:, :, c]
                if self.args.disableOpticalFlow == 1:
                    imagePixelData[i, c + 3] = torch.mul(imagePixelData[i, c + 3], 0)

        return imagePixelData

    # index contents of data: data[person_id][camera_id][img_id][channel_id][img_tensor_data]
    def prepareDataset(self, seq_root_rgb, seq_root_of, file_suffix):
        personDir = self.getPersonDirsList(seq_root_rgb)
        dataset = {}
        for i, pdir in enumerate(personDir):  # i=0, pdir='person_0001'
            dataset[i] = {}
            for cam in [1, 2]:
                if self.args.dataset == 1 or self.args.dataset == 0:
                    camera_name = ''.join(['cam', str(cam)])
                elif self.args.dataset == 2:
                    camera_name = ''.join(['cam_', str(cam)])  # 原始数据集的名称应该是cam_a,cam_b,为了方便，直接对原始数据集修改cam_1,cam_2
                rgb_dir = '/'.join([seq_root_rgb, camera_name, pdir])  # '/home/ying/Desktop/Spatial-Temporal-Pooling-Networks-ReID-master/data/prid_2011/multi_shot//cam_1/person_0001'
                of_dir = '/'.join([seq_root_of, camera_name, pdir])  # '/home/ying/Desktop/Spatial-Temporal-Pooling-Networks-ReID-master/data/prid2011flow/cam_1/person_0001'
                img_list = self.getSequenceImages(rgb_dir, file_suffix)
                dataset[i][cam] = self.loadSequenceImages(rgb_dir, of_dir, img_list)
        return dataset
