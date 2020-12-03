from __future__ import print_function, absolute_import
import time

import torch
from torch.autograd import Variable

from .evaluation_metrics import accuracy
from .loss import TripletLoss
from .utils.meters import AverageMeter
import torch.nn.functional as F


class BaseTrainer(object):
    def __init__(self, model, criterion, InvNet=None):
        super(BaseTrainer, self).__init__()
        self.model = model
        self.criterion = criterion
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.InvNet = InvNet

    def train(self, epoch, data_loader, optimizer, tri_weight, adv_weight, mem_weight, print_freq=1):
        optimizer_Encoder, optimizer_Transfer, optimizer_Cam = optimizer
        self.Encoder, self.Transfer, self.CamDis = self.model

        self.Encoder.train()
        self.Transfer.train()
        self.CamDis.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses_id = AverageMeter()
        losses_tri = AverageMeter()
        losses_cam = AverageMeter()
        losses_s_cam = AverageMeter()
        losses_mem = AverageMeter()
        precisions = AverageMeter()

        end = time.time()
        for i, inputs in enumerate(data_loader):
            data_time.update(time.time() - end)

            inputs, _, pids, cams, domain = self._parse_data(inputs)
            loss_id, loss_tri, loss_cam, loss_s_cam, loss_mem, prec1 = self._forward(inputs, pids, cams, domain, epoch, step=i)
            
            losses_id.update(loss_id.item(), pids.size(0))
            losses_tri.update(loss_tri.item(), pids.size(0))
            losses_cam.update(loss_cam.item(), pids.size(0))
            losses_s_cam.update(loss_s_cam.item(), pids.size(0))
            losses_mem.update(loss_mem.item(), pids.size(0))
            precisions.update(prec1, pids.size(0))

            # if epoch > 10:
                # loss = loss_id + tri_weight * loss_tri
            if epoch < 3:
                loss = loss_id + tri_weight * loss_tri + adv_weight * loss_s_cam + 0 * loss_mem
            else:
                loss = loss_id + tri_weight * loss_tri + adv_weight * loss_s_cam + mem_weight * loss_mem
            optimizer_Transfer.zero_grad()
            optimizer_Encoder.zero_grad()
            loss.backward()
            optimizer_Transfer.step()
            optimizer_Encoder.step()
            
            loss = loss_cam
            optimizer_Cam.zero_grad()
            loss.backward()
            optimizer_Cam.step()

            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0:
                print('Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                    #   'Data {:.3f} ({:.3f})\t'
                      'ID {:.3f} ({:.3f})\t'
                      'Tri {:.3f} ({:.3f})\t'
                      'cam {:.3f} ({:.3f})\t'
                      'advcam {:.3f} ({:.3f})\t'
                      'mem {:.5f} ({:.5f})\t'
                      'Prec {:.2%} ({:.2%})\t'
                      .format(epoch, i + 1, len(data_loader),
                              batch_time.val, batch_time.avg,
                            #   data_time.val, data_time.avg,
                              losses_id.val, losses_id.avg,
                              losses_tri.val, losses_tri.avg,
                              losses_cam.val, losses_cam.avg,
                              losses_s_cam.val, losses_s_cam.avg,
                              losses_mem.val, losses_mem.avg,
                              precisions.val, precisions.avg))

    def _parse_data(self, inputs):
        raise NotImplementedError

    def _forward(self, inputs, pids):
        raise NotImplementedError


class Trainer(BaseTrainer):
    def _parse_data(self, inputs):
        imgs, fnames, pids, cams, domain = inputs
        inputs = imgs.to(self.device)
        pids = pids.to(self.device)
        cams = cams.to(self.device)
        domain = domain.to(self.device)
        return inputs, fnames, pids, cams, domain

    def _forward(self, inputs, pids, cams, domain, epoch, step):
        x_feature = self.Encoder(inputs)
        _, trans_feature = self.Transfer(x_feature.clone().detach()) # transfer feature
        s_outputs, s_feature = self.Transfer(x_feature)

        # id
        loss_id = self.criterion[0](s_outputs, pids)
        loss_tri = self.criterion[1](s_feature, pids)
        prec, = accuracy(s_outputs.data, pids.data)
        prec = prec[0]

        # domain(cam)
        c_outputs = self.CamDis(trans_feature.clone().detach())
        loss_cam = self.criterion[0](c_outputs, cams)
        # 信息熵
        outputs_cam_s = self.CamDis(trans_feature)
        loss_s_cam = -torch.mean(torch.log(F.softmax(outputs_cam_s + 1e-6))) 

        # newcam = torch.ones(cams.shape).long().cuda()
        # loss_s_cam = self.criterion[0](outputs_cam_s, newcam)

        # mem
        loss_mem = self.InvNet(trans_feature, pids, domain, epoch=epoch, step=step)

        return loss_id, loss_tri, loss_cam, loss_s_cam, loss_mem, prec
