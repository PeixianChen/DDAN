from __future__ import print_function, absolute_import
import time
from collections import OrderedDict
import pdb
from collections import defaultdict

import torch
import numpy as np

from .evaluation_metrics import cmc, mean_ap
from .utils.meters import AverageMeter

from torch.autograd import Variable
from .utils import to_torch
from .utils import to_numpy
import pdb

def fliplr(img):
    '''flip horizontal'''
    inv_idx = torch.arange(img.size(3)-1,-1,-1).long()  # N x C x H x W N:TensornSamples in minibatch, i.e., batchsize x nChannels x Height x Width
    img_flip = img.index_select(3,inv_idx)
    return img_flip

def extract_cnn_feature(model, inputs, output_feature=None):
    # encoder, transfer, _, pfeNet = model
    encoder, transfer, _ = model
    encoder.eval()
    transfer.eval()
    # pfeNet.eval()
    # inputs = to_torch(inputs)
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # inputs = inputs.to(device)
    with torch.no_grad():
        feature = encoder(inputs.cuda())
        outputs = transfer(feature, output_feature=output_feature)
        # outputs = pfeNet(outputs)
        # _, outputs = model[0](inputs, output_feature=output_feature)
        # outputs = model.module.base(inputs)

        # outputs = torch.FloatTensor(inputs.size(0),1280).zero_()
        # for i in range(2):
        #     if(i==1):
        #         inputs = fliplr(inputs)
        #     # inputs = 
        #     # Variable(inputs.cuda())
        #     o = model(inputs, output_feature='pool5')
        #     # o = model.module.base(inputs.cuda())
        #     f = o.data.cpu()
        #     outputs = outputs+f

        fnorm = torch.norm(outputs, p=2, dim=1, keepdim=True)
        outputs = outputs.div(fnorm.expand_as(outputs))
        outputs = outputs.data.cpu()
    return outputs


def extract_features(model, data_loader, print_freq=1, output_feature=None):
    # encoder, transfer, _, pfeNet = model
    encoder, transfer, _ = model
    encoder.eval()
    transfer.eval()
    # pfeNet.eval()
    
    batch_time = AverageMeter()
    data_time = AverageMeter()

    features = OrderedDict()
    labels = OrderedDict()

    end = time.time()
    for i, (imgs, fnames, pids, _, _) in enumerate(data_loader):
        data_time.update(time.time() - end)

        outputs = extract_cnn_feature(model, imgs, output_feature)
        for fname, output, pid in zip(fnames, outputs, pids):
            features[fname] = output
            labels[fname] = pid

        batch_time.update(time.time() - end)
        end = time.time()

        #if (i + 1) % print_freq == 0:
        #    print('Extract Features: [{}/{}]\t'
        #          'Time {:.3f} ({:.3f})\t'
        #          'Data {:.3f} ({:.3f})\t'
        #          .format(i + 1, len(data_loader),
        #                  batch_time.val, batch_time.avg,
        #                  data_time.val, data_time.avg))

    return features, labels
    
def pairwise_distance(query_features, gallery_features, query=None, gallery=None):
    x = torch.cat([query_features[f].unsqueeze(0) for f, _, _,_ in query], 0)
    y = torch.cat([gallery_features[f].unsqueeze(0) for f, _, _,_ in gallery], 0)
    m, n = x.size(0), y.size(0)
    x = x.view(m, -1)
    y = y.view(n, -1)
    dist = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(m, n) + \
            torch.pow(y, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    dist.addmm_(1, -2, x, y.t())
    return dist

def evaluate_all(distmat, query=None, gallery=None,
                 query_ids=None, gallery_ids=None,
                 query_cams=None, gallery_cams=None,
                 cmc_topk=(1, 5, 10, 20)):
    if query is not None and gallery is not None:
        query_ids = [pid for _, pid, _,_ in query]
        gallery_ids = [pid for _, pid, _,_ in gallery]
        query_cams = [cam for _, _, cam,_ in query]
        gallery_cams = [cam for _, _, cam,_ in gallery]
    else:
        assert (query_ids is not None and gallery_ids is not None
                and query_cams is not None and gallery_cams is not None)

    # Compute mean AP
    mAP = mean_ap(distmat, query_ids, gallery_ids, query_cams, gallery_cams)
    print('Mean AP: {:4.1%}'.format(mAP))

    # Compute CMC scores
    cmc_configs = {
        'market1501': dict(separate_camera_set=False,
                           single_gallery_shot=False,
                           first_match_break=False)}
    cmc_scores = {name: cmc(distmat, query_ids, gallery_ids,
                            query_cams, gallery_cams, **params)
                  for name, params in cmc_configs.items()}

    print('CMC Scores')
    for k in cmc_topk:
        print('  top-{:<4}{:12.1%}'
              .format(k, cmc_scores['market1501'][k - 1]))

    return cmc_scores['market1501'][0]

class Evaluator(object):
    def __init__(self, model):
        super(Evaluator, self).__init__()
        self.model = model

    def evaluate(self, query_loader, gallery_loader, query, gallery, output_feature=None, rerank=False):
        query_features, _ = extract_features(self.model, query_loader, 1, output_feature)
        gallery_features, _ = extract_features(self.model, gallery_loader, 1, output_feature)
        if rerank:
            distmat = reranking(query_features, gallery_features, query, gallery)
        else:
            distmat = pairwise_distance(query_features, gallery_features, query, gallery)
        return evaluate_all(distmat, query=query, gallery=gallery)

class Evaluator(object):
    def __init__(self, model):
        super(Evaluator, self).__init__()
        self.model = model

    def evaluate(self, query_loader, gallery_loader, query, gallery, output_feature=None, rerank=False):
        query_features, _ = extract_features(self.model, query_loader, 1, output_feature)
        gallery_features, _ = extract_features(self.model, gallery_loader, 1, output_feature)
        if rerank:
            distmat = reranking(query_features, gallery_features, query, gallery)
        else:
            distmat = pairwise_distance(query_features, gallery_features, query, gallery)
        return evaluate_all(distmat, query=query, gallery=gallery)
    def __init__(self, model):
        super(Evaluator, self).__init__()
        self.model = model

        self.num_folds = 10
        self.test_all = False
        print(f'[evaluate] num_folds = {self.num_folds}, test_all = {self.test_all}')


    def eval_viper(self, query_loader, gallery_loader, query_data, gallery_data, output_feature='pool5', seed=0):
        rs = np.random.RandomState(seed)
        # rs = np.random.RandomState()

        # 提取所有样本的 feature
        query_features, _ = extract_features(self.model, query_loader, 1, output_feature)
        all_query_features = list(query_features.values())

        gallery_features, _ = extract_features(self.model, gallery_loader, 1, output_feature)
        all_gallery_features = list(gallery_features.values())

        # 基本数据
        _, all_query_pids, all_query_cids, _ = map(np.array, zip(*query_data))
        _, all_gallery_pids, all_gallery_cids, _ = map(np.array, zip(*gallery_data))

        q_pid_to_idx = defaultdict(list)
        for i, pid in enumerate(all_query_pids):
            q_pid_to_idx[pid].append(i)
        g_pid_to_idx = defaultdict(list)
        for i, pid in enumerate(all_gallery_pids):
            g_pid_to_idx[pid].append(i)

        # 多次测试
        num_tests = 632 if self.test_all else 316
        q_num_unique_pids = 632
        all_mAP, all_CMC, all_CMC5, all_CMC10 = [], [], [], []
        
        for j in range(self.num_folds):
            if j % 2 == 0:
                selected_pids = rs.choice(q_num_unique_pids, num_tests, replace=False) + 1
            else:
                new_selected_pids = [i for i in range(1, q_num_unique_pids+1) if i not in selected_pids]
                selected_pids = new_selected_pids
            gallery_idx, query_idx = [], []
            for pid in selected_pids:
                q_idx = q_pid_to_idx[pid][0]
                query_idx.append(q_idx)
            for pid in selected_pids:
                g_idx = g_pid_to_idx[pid][0]
                gallery_idx.append(g_idx)
            # 随机选取 num_tests 个样本测试
            selected_pids = rs.choice(q_num_unique_pids, num_tests, replace=False) + 1
            #selected_pids = sorted(selected_pids)

            # # 划分 gallery 和 query，即每个 pid 选 2 个样本
            # if j % 2 ==0:
            #     gallery_idx, query_idx = [], []
            #     for pid in selected_pids:
            #         q_idx = q_pid_to_idx[pid][0]
            #         query_idx.append(q_idx)
            #     for pid in selected_pids:
            #         g_idx = g_pid_to_idx[pid][0]
            #         gallery_idx.append(g_idx)
            # else:
            #     gallery_idx, query_idx = query_idx, gallery_idx

            def get(x, idx):
                return [x[i] for i in idx]

            # 获取 gallery
            gallery_features = torch.stack(get(all_gallery_features, gallery_idx))
            gallery_pids = get(all_gallery_pids, gallery_idx)
            gallery_cids = get(all_gallery_cids, gallery_idx)

            # 获取 query
            query_features = torch.stack(get(all_query_features, query_idx))
            query_pids = get(all_query_pids, query_idx)
            query_cids = get(all_query_cids, query_idx)

            # 测试
            dist = self._pdist(query_features, gallery_features)
            #dist = self._pdist(gallery_features, query_features)
            mAP, CMC, CMC5, CMC10 = self._eval_dist(dist, gallery_pids, gallery_cids, query_pids, query_cids)
            acc = self._simple_acc(dist, gallery_pids, query_pids)
            all_mAP.append(mAP)
            all_CMC.append(CMC)
            all_CMC5.append(CMC5)
            all_CMC10.append(CMC10)
            # all_acc.append(acc)

        print('VIPeR')
        print(f'[map] {np.mean(all_mAP):.2%} |', ' '.join(map('{:.2%}'.format, sorted(all_mAP))))
        print(f'[cmc] {np.mean(all_CMC):.2%} |', ' '.join(map('{:.2%}'.format, sorted(all_CMC))))
        print("5:", np.mean(all_CMC5))
        print("10:", np.mean(all_CMC10))
        #print(f'[acc] {np.mean(all_acc):.2%} |', ' '.join(map('{:.2%}'.format, sorted(all_acc)))) # acc == cmc(top-1)
        return np.mean(all_CMC)
    def eval_grid(self, query_loader, gallery_loader, query_data, gallery_data, output_feature='pool5', seed=0):
        rs = np.random.RandomState(seed)
        # 提取所有样本的 feature
        query_features, _ = extract_features(self.model, query_loader, 1, output_feature)
        all_query_features = list(query_features.values())

        gallery_features, _ = extract_features(self.model, gallery_loader, 1, output_feature)
        all_gallery_features = list(gallery_features.values())

        # 基本数据
        _, all_query_pids, all_query_cids, _ = map(np.array, zip(*query_data))
        _, all_gallery_pids, all_gallery_cids, _ = map(np.array, zip(*gallery_data))

        q_pid_to_idx = defaultdict(list)
        for i, pid in enumerate(all_query_pids):
            q_pid_to_idx[pid].append(i)
        g_pid_to_idx = defaultdict(list)
        for i, pid in enumerate(all_gallery_pids):
            g_pid_to_idx[pid].append(i)

        # 多次测试
        num_tests = 250 if self.test_all else 125
        q_num_unique_pids = 250
        all_mAP, all_CMC, all_CMC5, all_CMC10 = [], [], [], []
        
        for _ in range(self.num_folds):
            # 随机选取 num_tests 个样本测试
            selected_pids = rs.choice(q_num_unique_pids, num_tests, replace=False) + 1

            # 划分 gallery 和 query，即每个 pid 选 2 个样本
            gallery_idx, query_idx = [], []
            for pid in selected_pids:
                q_idx = q_pid_to_idx[pid][0]
                query_idx.append(q_idx)
            gallery_idx = gallery_idx + g_pid_to_idx[0]
            for pid in selected_pids:
                g_idx = g_pid_to_idx[pid][0]
                gallery_idx.append(g_idx)

            def get(x, idx):
                return [x[i] for i in idx]

            # 获取 gallery
            gallery_features = torch.stack(get(all_gallery_features, gallery_idx))
            gallery_pids = get(all_gallery_pids, gallery_idx)
            gallery_cids = get(all_gallery_cids, gallery_idx)

            # 获取 query
            query_features = torch.stack(get(all_query_features, query_idx))
            query_pids = get(all_query_pids, query_idx)
            query_cids = get(all_query_cids, query_idx)

            # 测试
            dist = self._pdist(query_features, gallery_features)
            mAP, CMC, CMC5, CMC10 = self._eval_dist(dist, gallery_pids, gallery_cids, query_pids, query_cids)
            all_mAP.append(mAP)
            all_CMC.append(CMC)
            all_CMC5.append(CMC5)
            all_CMC10.append(CMC10)

        print('GRID')
        print(f'[map] {np.mean(all_mAP):.2%} |', ' '.join(map('{:.2%}'.format, sorted(all_mAP))))
        print(f'[cmc] {np.mean(all_CMC):.2%} |', ' '.join(map('{:.2%}'.format, sorted(all_CMC))))
        print("5:", np.mean(all_CMC5))
        print("10:", np.mean(all_CMC10))
        return np.mean(all_CMC)
    def eval_prid(self, query_loader, gallery_loader, query_data, gallery_data, output_feature='pool5', seed=0):
        rs = np.random.RandomState(seed)
        # 提取所有样本的 feature
        query_features, _ = extract_features(self.model, query_loader, 1, output_feature)
        all_query_features = list(query_features.values())

        gallery_features, _ = extract_features(self.model, gallery_loader, 1, output_feature)
        all_gallery_features = list(gallery_features.values())

        # 基本数据
        _, all_query_pids, all_query_cids, _ = map(np.array, zip(*query_data))
        _, all_gallery_pids, all_gallery_cids, _ = map(np.array, zip(*gallery_data))

        q_pid_to_idx = defaultdict(list)
        for i, pid in enumerate(all_query_pids):
            q_pid_to_idx[pid].append(i)
        g_pid_to_idx = defaultdict(list)
        for i, pid in enumerate(all_gallery_pids):
            g_pid_to_idx[pid].append(i)

        # 多次测试
        num_tests = 200 if self.test_all else 100
        q_num_unique_pids = 200
        all_mAP, all_CMC, all_CMC5, all_CMC10 = [], [], [], []
        
        for _ in range(self.num_folds):
            # 随机选取 num_tests 个样本测试
            selected_pids = rs.choice(q_num_unique_pids, num_tests, replace=False) + 1

            # 划分 gallery 和 query，即每个 pid 选 2 个样本
            gallery_idx, query_idx = [], []
            for pid in selected_pids:
                q_idx = q_pid_to_idx[pid][0]
                query_idx.append(q_idx)
            for pid in selected_pids.tolist() + [i for i in range(201, 750)]:
                g_idx = g_pid_to_idx[pid][0]
                gallery_idx.append(g_idx)
            
            def get(x, idx):
                return [x[i] for i in idx]

            # 获取 gallery
            gallery_features = torch.stack(get(all_gallery_features, gallery_idx))
            gallery_pids = get(all_gallery_pids, gallery_idx)
            gallery_cids = get(all_gallery_cids, gallery_idx)

            # 获取 query
            query_features = torch.stack(get(all_query_features, query_idx))
            query_pids = get(all_query_pids, query_idx)
            query_cids = get(all_query_cids, query_idx)

            # 测试
            dist = self._pdist(query_features, gallery_features)
            mAP, CMC, CMC5, CMC10 = self._eval_dist(dist, gallery_pids, gallery_cids, query_pids, query_cids)
            all_mAP.append(mAP)
            all_CMC.append(CMC)
            all_CMC5.append(CMC5)
            all_CMC10.append(CMC10)

        print('PRID')
        print(f'[map] {np.mean(all_mAP):.2%} |', ' '.join(map('{:.2%}'.format, sorted(all_mAP))))
        print(f'[cmc] {np.mean(all_CMC):.2%} |', ' '.join(map('{:.2%}'.format, sorted(all_CMC))))
        print("5:", np.mean(all_CMC5))
        print("10:", np.mean(all_CMC10))
        return np.mean(all_CMC)
    def eval_ilids(self, all_loader, all_data, output_feature=None, seed=0):
        """
        Args:
          all_loader: loader of iLids inputs
          all_data: list of labels, (fname, person_id, cam_id, ?)
          output_feature: ?
          seed: random seed only for eval

        Note:
          loader 和 data 应该是按顺序的，[id1.cam1/2/3, id2.cam1/2/3, ....]
          不按顺序应该也可以
        """
        # 这个随机状态只在 eval 中使用，不影响外部的随机状态
        rs = np.random.RandomState(seed)

        # 提取所有样本的 feature
        features, _ = extract_features(self.model, all_loader, 1, output_feature)
        features = list(features.values())

        # 基本数据
        _, all_pids, all_cids, _ = map(np.array, zip(*all_data))
        num_unique_pids = len(set(all_pids))

        pid_to_idx = defaultdict(list)
        for i, pid in enumerate(all_pids):
            pid_to_idx[pid].append(i)

        # 多次测试
        num_tests = 119 if self.test_all else 60
        all_mAP, all_CMC, all_CMC5, all_CMC10 = [], [], [], []
        for _ in range(self.num_folds):
            # 随机选取 num_tests 个样本测试
            selected_pids = rs.choice(num_unique_pids, num_tests, replace=False) + 1

            # 划分 gallery 和 query，即每个 pid 选 2 个样本
            gallery_idx, query_idx = [], []
            for pid in selected_pids:
                idx1, idx2 = rs.choice(pid_to_idx[pid], 2, replace=False)
                gallery_idx.append(idx1)
                query_idx.append(idx2)

            def get(x, idx):
                return [x[i] for i in idx]

            # 获取 gallery
            gallery_features = torch.stack(get(features, gallery_idx))
            gallery_pids = get(all_pids, gallery_idx)
            gallery_cids = get(all_cids, gallery_idx)

            # 获取 query
            query_features = torch.stack(get(features, query_idx))
            query_pids = get(all_pids, query_idx)
            query_cids = get(all_cids, query_idx)

            # 测试
            dist = self._pdist(query_features, gallery_features)
            mAP, CMC, CMC5, CMC10 = self._eval_dist(dist, gallery_pids, gallery_cids, query_pids, query_cids)
            all_mAP.append(mAP)
            all_CMC.append(CMC)
            all_CMC5.append(CMC5)
            all_CMC10.append(CMC10)

        print('iLIDs')
        print(f'[map] {np.mean(all_mAP):.2%} |', ' '.join(map('{:.2%}'.format, sorted(all_mAP))))
        print(f'[cmc] {np.mean(all_CMC):.2%} |', ' '.join(map('{:.2%}'.format, sorted(all_CMC))))
        print("5:", np.mean(all_CMC5))
        print("10:", np.mean(all_CMC10))
        return np.mean(all_CMC)
    @staticmethod
    # def _pdist(input1, input2):
        # dist = 1 - torch.mm(input1, input2.t())
        # dist = 1 - torch.cosine_similarity(input1,input2, dim=1)
        # return dist
    def _pdist(x, y):
        m, n = x.size(0), y.size(0)
        x, y = x.view(m, -1), y.view(n, -1)
        xx = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(m, n)
        yy = torch.pow(y, 2).sum(dim=1, keepdim=True).expand(n, m).t()
        xy = x @ y.t()
        dist = xx - 2 * xy + yy
        return dist

    @staticmethod
    def _eval_dist(dist, gallery_pids, gallery_cids, query_pids, query_cids):
        args = (dist, query_pids, gallery_pids, query_cids, gallery_cids)
        kwargs = dict(separate_camera_set=False, single_gallery_shot=False, first_match_break=False)
        mAP, CMC = mean_ap(*args), cmc(*args, **kwargs)
        return mAP, CMC[0], CMC[1], CMC[5]

    @staticmethod
    def _simple_acc(dist, gallery_pids, query_pids):
        pred = dist.argmin(1)
        total = len(query_pids)
        good = sum([gallery_pids[pred[i]] == query_pids[i] for i in range(pred.shape[0])])
        return 1. * good / total


