from __future__ import print_function, absolute_import
import os.path as osp
import numpy as np
import pdb
from glob import glob
import re


class DA(object):

    def __init__(self, data_dir):

        # training image dir
        self.market_images_dir = osp.join(data_dir, "DG/market1501")
        self.duke_images_dir = osp.join(data_dir, "DG/DukeMTMC-reID")
        self.sysu_images_dir = osp.join(data_dir, "DG/CUHK-SYSU")
        self.cuhk03_images_dir = osp.join(data_dir, "DG/cuhk03")
        self.cuhk02_images_dir = osp.join(data_dir, "DG/cuhk02")

        self.source_train_path = 'bounding_box_train'
        self.load()

    def preprocess(self, images_dir, path, relabel=True):
        add_pid = 0
        add_cam = 0
        domain_all = 0
        domain = 0

        if images_dir == self.duke_images_dir:
            add_pid = 1501
            add_cam = 10
            domain_all = 1
            domain = 0
        if images_dir == self.cuhk03_images_dir:
            add_pid = 1501+1812
            add_cam = 10+2
            domain_all = 2
            domain = 0
        if images_dir == self.cuhk02_images_dir:
            add_pid = 1501+1812+1467
            add_cam = 10+2+8
            domain_all = 3
            domain = 0
        if images_dir == self.sysu_images_dir:
            add_pid = 1501+1816+1467+1812
            add_cam = 10+2+8+6
            domain_all = 4
            domain = 1

        pattern = re.compile(r'([-\d]+)_c?(\d)')
        all_pids = {}
        ret = []
        fpaths = sorted(glob(osp.join(images_dir, path, '*.jpg')))
        type = ['*jpeg', '*.png', '*bmp']
        t = 0
        while fpaths == []:
            print(osp.join(images_dir, path, type[t]))
            fpaths = sorted(glob(osp.join(images_dir, path, type[t])))
            t += 1
        for fpath in fpaths:
            # fname = osp.basename(fpath)
            fname = fpath
            pid, cam = map(int, pattern.search(fname).groups())
            if pid == -1: continue
            if relabel:
                if pid not in all_pids:
                    all_pids[pid] = len(all_pids)
            else:
                if pid not in all_pids:
                    all_pids[pid] = pid
            pid = all_pids[pid]
            cam -= 1
            ret.append((fname, pid+add_pid, domain, domain_all))
        return ret, int(len(all_pids))

    def load(self):
        self.market_train, self.num_market_ids = self.preprocess(self.market_images_dir, self.source_train_path)
        self.duke_train, self.num_duke_ids = self.preprocess(self.duke_images_dir, self.source_train_path)
        self.sysu_train, self.num_sysu_ids = self.preprocess(self.sysu_images_dir, self.source_train_path)
        self.cuhk03_train, self.num_cuhk03_ids = self.preprocess(self.cuhk03_images_dir, self.source_train_path)
        self.cuhk02_train, self.num_cuhk02_ids = self.preprocess(self.cuhk02_images_dir, self.source_train_path)

        unique_ids = set()
        all_dataset = [self.market_train, self.duke_train, self.cuhk03_train, self.cuhk02_train, self.sysu_train]
        for dataset in all_dataset:
            ids = set(i for _, i, _, _ in dataset)
            assert not unique_ids & ids
            unique_ids |= ids

        self.source_train = self.market_train + self.duke_train  + self.cuhk03_train + self.cuhk02_train + self.sysu_train
        self.num_source_ids = self.num_market_ids + self.num_duke_ids + self.num_cuhk03_ids + self.num_cuhk02_ids + self.num_sysu_ids

        print(self.__class__.__name__, "dataset loaded")
        print("  subset          |  # ids  | # images")
        print("  ------------------------------------")
        print("  market train    |  {:5d}  | {:8d}"
              .format(self.num_market_ids, len(self.market_train)))
        print("  duke train      | {:5d} | {:8d}"
              .format(self.num_duke_ids, len(self.duke_train)))
        print("  cuhk03 train    | {:5d} | {:8d}"
              .format(self.num_cuhk03_ids, len(self.cuhk03_train)))
        print("  sysu train      | {:5d} | {:8d}"
              .format(self.num_sysu_ids, len(self.sysu_train)))
        print("  cuhk02 train    | {:5d} | {:8d}"
              .format(self.num_cuhk02_ids, len(self.cuhk02_train)))
        print("  total train     | {:5d} | {:8d}"
              .format(self.num_source_ids, len(self.source_train)))
