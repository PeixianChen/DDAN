from __future__ import absolute_import
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data.sampler import (
    Sampler, SequentialSampler, RandomSampler, SubsetRandomSampler,
    WeightedRandomSampler)

import copy
import random

import torch
from collections import defaultdict

import numpy as np


class RandomIdentitySampler(Sampler):
    """
    Randomly sample N identities, then for each identity,
    randomly sample K instances, therefore batch size is N*K.
    Args:
    - data_source (list): list of (img_path, pid, camid).
    - num_instances (int): number of instances per identity in a batch.
    - batch_size (int): number of examples in a batch.
    """

    def __init__(self, data_source, batch_size, num_instances):
        self.data_source = data_source
        self.batch_size = batch_size
        self.num_instances = num_instances
        self.num_pids_per_batch = self.batch_size // self.num_instances
        self.index_dic = defaultdict(list)
        for index, (_, pid, _, _) in enumerate(self.data_source):
            self.index_dic[pid].append(index)
        self.pids = list(self.index_dic.keys())

        # estimate number of examples in an epoch
        self.length = 0
        for pid in self.pids:
            idxs = self.index_dic[pid]
            num = len(idxs)
            if num < self.num_instances:
                num = self.num_instances
            self.length += num - num % self.num_instances

    def __iter__(self):
        batch_idxs_dict = defaultdict(list)

        for pid in self.pids:
            idxs = copy.deepcopy(self.index_dic[pid])
            if len(idxs) < self.num_instances:
                idxs = np.random.choice(idxs, size=self.num_instances, replace=True)
            random.shuffle(idxs)
            batch_idxs = []
            for idx in idxs:
                batch_idxs.append(idx)
                if len(batch_idxs) == self.num_instances:
                    batch_idxs_dict[pid].append(batch_idxs)
                    batch_idxs = []

        avai_pids = copy.deepcopy(self.pids)
        final_idxs = []

        while len(avai_pids) >= self.num_pids_per_batch:
            selected_pids = random.sample(avai_pids, self.num_pids_per_batch)
            for pid in selected_pids:
                batch_idxs = batch_idxs_dict[pid].pop(0)
                final_idxs.extend(batch_idxs)
                if len(batch_idxs_dict[pid]) == 0:
                    avai_pids.remove(pid)

        self.length = len(final_idxs)
        return iter(final_idxs)

    def __len__(self):
        return self.length
# class RandomIdentitySampler(Sampler):
#     def __init__(self, data_source, num_instances=1):
#         self.data_source = data_source
#         self.num_instances = num_instances
#         self.index_dic = defaultdict(list)
#         for index, (_, pid, _) in enumerate(data_source):
#             self.index_dic[pid].append(index)
#         self.pids = list(self.index_dic.keys())
#         self.num_samples = len(self.pids)

#     def __len__(self):
#         return self.num_samples * self.num_instances

#     def __iter__(self):
#         indices = torch.randperm(self.num_samples)
#         ret = []
#         for i in indices:
#             pid = self.pids[i]
#             t = self.index_dic[pid]
#             if len(t) >= self.num_instances:
#                 t = np.random.choice(t, size=self.num_instances, replace=False)
#             else:
#                 t = np.random.choice(t, size=self.num_instances, replace=True)
#             ret.extend(t)
#         return iter(ret)

class IdentitySampler(Sampler):
    """Sample person identities evenly in each batch.
        Args:
            train_color_label, train_thermal_label: labels of two modalities
            color_pos, thermal_pos: positions of each identity
            batchSize: batch size
    """

    def __init__(self, data_source, num_instances=1):  
        self.index_dic = defaultdict(list)
        for index, (_, pid, _) in enumerate(data_source):
            self.index_dic[pid].append(index)
        self.pids = np.array(list(self.index_dic.keys()))
        # uni_label = np.unique(train_color_label)
        self.n_classes = len(self.pids)
        # sample_thermal = np.arange(batchSize)
        self.N = len(data_source)
        self.num_instances = num_instances

        self.ret = []
        
        for pid, pid_indexes in self.index_dic.items():
            if len(pid_indexes) % self.num_instances != 0:
                r = np.random.choice(pid_indexes, len(pid_indexes) % self.num_instances, replace=False)
                pid_indexes += r.tolist()
            random.shuffle(pid_indexes)
            for i in range(0, len(pid_indexes), self.num_instances):
                self.ret.append(pid_indexes[i:i+self.num_instances])
        random.shuffle(self.ret)
        self.ret = [i for j in self.ret for i in j]
        
        
    def __iter__(self):
        self.ret = []
        for pid, pid_indexes in self.index_dic.items():
            if len(pid_indexes) % self.num_instances != 0:
                r = np.random.choice(pid_indexes, len(pid_indexes) % self.num_instances, replace=False)
                pid_indexes += r.tolist()
            random.shuffle(pid_indexes)
            for i in range(0, len(pid_indexes), self.num_instances):
                self.ret.append(pid_indexes[i:i+self.num_instances])
        random.shuffle(self.ret)
        self.ret = [i for j in self.ret for i in j]
        return iter(self.ret)

    def __len__(self):
        return len(self.ret)
