import torch
import torch.nn.functional as F
from torch import nn, autograd
from torch.autograd import Variable, Function
import numpy as np
import math

from .wassdistance import SinkhornDistance

class ExemplarMemory(Function):
    def __init__(self, em, em_cnt, alpha=0.01):
        super(ExemplarMemory, self).__init__()
        self.em = em
        self.em_cnt = em_cnt # 记录当前 em 每个 entry 是多少个样本的 mean
        self.alpha = alpha

    def forward(self, inputs, targets):
        self.save_for_backward(inputs, targets)
        if self.em.sum() == 0:
            outputs = inputs.mm(self.em.t())
        else:
            em = self.em / self.em.norm(p=2, dim=1, keepdim=True)
            outputs = inputs.mm(em.t())
        return outputs

    def backward(self, grad_outputs):
        inputs, targets = self.saved_tensors
        grad_inputs = None
        if self.needs_input_grad[0]:
            grad_inputs = grad_outputs.mm(self.em)
        # for x, y in zip(inputs, targets):
        #     n = self.em_cnt[y]
        #     n += 1
        #     self.em[y] = self.em[y] * (n - 1) / n + x / n

        n = self.em_cnt[targets]
        n += 1
        self.em[y] = (self.em[y] * (n-1) + x) / n

        return grad_inputs, None

class InvNet(nn.Module):
    def __init__(self, num_features, num_classes, batchsize, beta=0.05, knn=6, alpha=0.01):
        super(InvNet, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_features = num_features
        self.num_classes = num_classes
        self.alpha = alpha  # Memory update rate
        self.beta = beta  # Temperature fact
        self.knn = knn  # Knn for neighborhood invariance

        # Exemplar memory
        self.em = nn.Parameter(torch.zeros(num_classes, num_features), requires_grad=False)
        self.em_cnt = torch.zeros(num_classes)
        self.untouched_targets = set(range(num_classes))
        self.em_current = torch.zeros(num_classes, num_features).cuda()
        self.em_last = torch.zeros(num_classes, num_features).cuda()
        self.domain = torch.from_numpy(np.array([0 for _ in range(1816)] + [1 for _ in range(1467)] + [2 for _ in range(1812)] + [3 for _ in range(1501)] + [4 for i in range(11934)])).unsqueeze(0).cuda()
        self.domain = self.domain.repeat(batchsize, 1)

        self.id = [[] for _ in range(18530)]

    def forward(self, inputs, label, domain, epoch=None, step=None, fnames_target=None):
        '''
        inputs: [128, 2048], each t's 2048-d feature
        label: [128], each t's label
        '''
        if step == 0:
            print(self.em)
        # alpha = self.alpha * epoch
        alpha = self.alpha
        if epoch ==0 :
            alpha = 0
        if epoch > 0 and step==0:
            # self.em_cnt = self.em_cnt * 0 + 2

            self.em_last = self.em.clone()
            self.em_cnt = self.em_cnt * 0
        if epoch > 0:
            em = self.em / self.em.norm(p=2, dim=1, keepdim=True)
        else:
            em = self.em
    
        tgt_feature = inputs.mm(em.t())
        tgt_feature /= 0.05


        loss = self.smooth_loss(self.em, inputs, tgt_feature, label, domain, epoch)

        for x, y in zip(inputs, label):
            n = self.em_cnt[y]
            n += 1
            # self.em.data[y] = self.em.data[y] * (n - 1) / n + x.data / n
            
            self.em_current[y] = self.em_current[y] * (n - 1) / n + x.data / n
            self.em.data[y] = self.em_last[y] * alpha + self.em_current[y] * (1-alpha)
        return loss

    def smooth_loss(self, em, inputs_feature, tgt_feature, label, domain, epoch):
        '''
        tgt_feature: [128, 16522], similarity of batch & targets
        label: see forward
        '''
        mask = self.smooth_hot(tgt_feature.detach().clone(), label.detach().clone(), self.knn, domain)

        # batchsize是64
        new_feature = []
        for m in mask:
            index = m.nonzero()
            new_feature.append(em[index])
        new_feature = torch.cat(new_feature,0).squeeze(1)
        # inputs_feature = torch.cat([inputs_feature for _ in range(self.knn*4)],1)
        # inputs_feature = torch.reshape(inputs_feature,(64*self.knn*4, 1280))


        # ------------12.30------------------------
        # 如果domain 是sysu, 则不和其他对比，为0
        # mask_sysu = domain.clone()
        # mask_sysu[domain != 4] = 1
        # mask_sysu[domain == 4] = 0

        # inputs_feature = inputs_feature * mask_sysu.unsqueeze(1)
        # 为了消除inputs下来的梯度
        # ------------12.30------------------------

        inputs_feature = torch.cat([inputs_feature for _ in range(self.knn)],1)
        # batchsize=64
        inputs_feature = torch.reshape(inputs_feature,(64*self.knn, 1280))
        if epoch > 0:
            inputs_feature = F.softmax(inputs_feature / self.beta, dim=1)
            new_feature = F.softmax(new_feature / self.beta, dim=1)
        # print(inputs_feature)
        # print(new_feature)
        loss = (nn.KLDivLoss()(torch.log(inputs_feature + 1e-8), new_feature) + nn.KLDivLoss()(torch.log(new_feature + 1e-8), inputs_feature))
        # print(loss)
        # print(a)

        # outputs = F.log_softmax(tgt_feature, dim=1)
        # loss = - (mask * outputs)
        # loss = loss.sum(dim=1)
        # loss = loss.mean(dim=0)

        # wassdistance
        # sinkhorn = SinkhornDistance(eps=0.1, max_iter=100, reduction=None)
        # loss, _, _ = sinkhorn(inputs_feature, new_feature)

        return loss

    def smooth_hot(self, tgt_feature, targets, k=6, domain=None):
        '''
        see smooth_loss
        '''
        mask = torch.zeros(tgt_feature.size()).to(self.device)

        # 如果domain 是sysu, 则不和其他对比，为0
        # mask_sysu = domain.clone()
        # mask_sysu[domain != 4] = 1
        # mask_sysu[domain == 4] = 0
        
        # 得到的mask_d：自己的domain为0, 与自己不同的为1
        domain = domain.unsqueeze(1).repeat(1,18530)
        mask_d = domain-self.domain
        mask_d[mask_d!=0] = 1
        # print(mask_d)
        # print(mask_d[mask_d == 0].shape)
        # mask = 1 - mask

        #----------------12.30---------------
        # SYSU找自己域内的拉近
        mask_d[domain == 4] = 1 - mask_d[domain == 4]
        #----------------12.30---------------

        # # 在memorybank中，对于每个domain都找到k个
        # _feature = tgt_feature * mask_d
        # # 第1个domain,ID数量1816
        # _, topk = tgt_feature[:,:1816].topk(k, dim=1)
        # mask.scatter_(1, topk, 1)

        # # 第2个domain,1467
        # _, topk = tgt_feature[:,1816:1816+1467].topk(k, dim=1)
        # mask.scatter_(1, topk+1816, 1)

        # # 第3个domain,ID数量1812
        # _, topk = tgt_feature[:,1816+1467:1816+1467+1812].topk(k, dim=1)
        # mask.scatter_(1, topk+1816+1467, 1)

        # # 第4个domain,ID数量1501
        # _, topk = tgt_feature[:,1816+1467+1812:1816+1467+1812+1501].topk(k, dim=1)
        # mask.scatter_(1, topk+1816+1467+1812, 1)

        # # 第5个domain,ID数量11934
        # _, topk = tgt_feature[:,1816+1467+1812+1501:].topk(k, dim=1)
        # mask.scatter_(1, topk+1816+1467+1812+1501, 1)
        # # 这里得到的mask:共有4*k个1,(自身domain内的ID都为0)
        # mask = mask * mask_d.float()

        _feature = tgt_feature * mask_d  #* mask_sysu.unsqueeze(1)
        _, topk = _feature.topk(k, dim=1)
        # print("different----------")
        # for i, k in enumerate(topk):
        #             if domain[i][0] == 2:
        #         #     if targets[i] == 4459:
        #                 print(domain[i][0].item(), targets[i], k[0].item(), tgt_feature[i][8983])
        #         #         # print(a)
        # print("different----------")
        # # 找同domain里相似度相似度比较低，但是像的：
        # mask_d = 1 - mask_d
        # _, topk = (tgt_feature * mask_d).topk(80, dim=1)
        # print("same----------")
        # for i, k in enumerate(topk):
        #             if domain[i][0] == 2:
        #                 print(tgt_feature[i][k[0].item()])
        #                 print(domain[i][0].item(), targets[i], k[-1].item(), tgt_feature[i][3667])

        # print("same----------")

        mask.scatter_(1, topk, 1)

        # print(mask.sum())
        # # 自己
        # index_2d = targets[..., None]
        # mask.scatter_(1, index_2d, 3)

        return mask
