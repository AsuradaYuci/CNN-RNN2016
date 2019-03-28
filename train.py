#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# @author = yuci
# Date: 19-3-25 下午8:13


from __future__ import print_function, absolute_import
import time
import torch
from torch import nn
from eval.eva_functions import accuracy
from utils import AverageMeter
import torch.nn.functional as F
from utils import to_numpy
# from tensorboardX import SummaryWriter
# writer = SummaryWriter('/media/ying/0BDD17830BDD1783/video_reid/logs/generate_q_by_lstm_100epoch')
# mode decide how to train the model
"""  分割损失回传
            optimizer1.zero_grad()
            loss1.backward(retain_graph=True)
            optimizer1.step()

            optimizer2.zero_grad()
            loss2.backward()
            optimizer2.step()
"""
"""  累积梯度回传
            # if (i + 1) % accumulation_steps == 0:
            #     optimizer1.step()
            #     optimizer2.step()
            #     optimizer1.zero_grad()
            #     optimizer2.zero_grad()
            #     # loss.backward()
"""


class BaseTrainer(object):

    def __init__(self, model, criterion):
        super(BaseTrainer, self).__init__()
        self.model = model
        self.criterion = criterion
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def train(self, epoch, data_loader, optimizer):
        self.model.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        precisions = AverageMeter()
        precisions1 = AverageMeter()
        accumulation_steps = 8
        # total_loss = 0

        end = time.time()
        for i, inputs in enumerate(data_loader):
            data_time.update(time.time() - end)

            netinputs0, netinputs1, targets = self._parse_data(inputs)

            loss = self._forward(netinputs0, netinputs1, targets)  # 1.前向传播
            # loss = loss1 + loss2
            # total_loss += loss.item()
            losses.update(loss.item(), len(targets[0]))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_time.update(time.time() - end)
            end = time.time()
            print_freq = 10
            if (i + 1) % print_freq == 0:
                print('Epoch: [{}][{}/{}]\t'
                      'Loss {:.3f} ({:.3f})\t'
                      .format(epoch, i + 1, len(data_loader),
                              losses.val, losses.avg))

    def _parse_data(self, inputs):
        raise NotImplementedError

    def _forward(self, netinputs0, netinputs1, targets):
        raise NotImplementedError


class SEQTrainer(BaseTrainer):

    def __init__(self, model, criterion):
        super(SEQTrainer, self).__init__(model, criterion)
        self.criterion = criterion
        # self.rate = rate
        # self.softmax = nn.Softmax()

    def _parse_data(self, inputs):
        seq0, seq1, targets = inputs
        seq0 = seq0.to(self.device)
        seq1 = seq1.to(self.device)

        return seq0, seq1, targets

    def _forward(self, netinputs0, netinputs1, targets):
# todo: target有问题？
        feature_p, feature_g, identity_p, identity_g = self.model(netinputs0, netinputs1)
        loss = self.criterion(feature_p, feature_g, identity_p, identity_g, targets)

        return loss

    def train(self, epoch, data_loader, optimizer):
        # self.rate = rate
        super(SEQTrainer, self).train(epoch, data_loader, optimizer)
