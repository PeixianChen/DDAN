from __future__ import print_function, absolute_import
import os.path as osp
import numpy as np
import pdb
from glob import glob
import re
import random


class TotalData(object):

    def __init__(self, data_dir):

        self.prid_images_dir = osp.join(data_dir, "DG/PRID")
        self.grid_images_dir = osp.join(data_dir, "DG/GRID")
        self.viper_images_dir = osp.join(data_dir, "DG/VIPeR")
        self.ilid_images_dir = osp.join(data_dir, "DG/iLIDS")

        # training image dir
        self.gallery_path = 'gallery_test'
        self.query_path = 'query_test'
        self.load()

    def preprocess(self, images_dir, path, relabel=True):
        domain = 5
        pattern = re.compile(r'([-\d]+)_c?(\d)')
        all_pids = {}
        ret = []
        fpaths = sorted(glob(osp.join(images_dir, path, '*.jpg')))
        type = ['*jpeg', '*.png', '*bmp']
        t = 0
        while fpaths == []:
            fpaths = sorted(glob(osp.join(images_dir, path, type[t])))
            t += 1
        for fpath in fpaths:
            fname = osp.basename(fpath)
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
            ret.append((fname, pid, cam, cam))
        return ret, int(len(all_pids))

    def ILIDS_preprocess(self, images_dir, path, relabel=True):
        domain = 5
        ilid_query, ilid_gallery = [], []

        pattern = re.compile(r'([-\d]+)_c?(\d)')
        all_pids = {}
        ret = []
        fpaths = glob(osp.join(images_dir, path, '*.jpg'))
        type = ['*jpeg', '*.png', '*bmp']
        t = 0
        while fpaths == []:
            fpaths = sorted(glob(osp.join(images_dir, path, type[t])))
            t += 1
        for fpath in fpaths:
            fname = osp.basename(fpath)
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
            ilid_query.append((fname, pid, cam, cam))
            ilid_gallery.append((fname, pid, cam,cam))
        return ilid_query, ilid_gallery, int(len(all_pids))

    def load(self):
        # DG TEST
        self.prid_gallery, self.num_gallery_ids = self.preprocess(self.prid_images_dir, self.gallery_path, False)
        self.prid_query, self.num_query_ids = self.preprocess(self.prid_images_dir, self.query_path, False)
        self.grid_gallery, self.num_gallery_ids = self.preprocess(self.grid_images_dir, self.gallery_path, False)
        self.grid_query, self.num_query_ids = self.preprocess(self.grid_images_dir, self.query_path, False)
        self.viper_gallery, self.num_gallery_ids = self.preprocess(self.viper_images_dir, self.gallery_path, False)
        self.viper_query, self.num_query_ids = self.preprocess(self.viper_images_dir, self.query_path, False)

        self.ilid_query, self.ilid_gallery, self.num_query_ids = self.ILIDS_preprocess(self.ilid_images_dir, "images", False)

        print(self.__class__.__name__, "dataset loaded")
        print("  subset          |  # query  | # gallery")
        print("  ------------------------------------")
        print("  prid train    |  {:8d}  | {:8d}  "
              .format(len(self.prid_query), len(self.prid_gallery)))
        print("  grid train    | {:8d} | {:8d} "
              .format(len(self.grid_query), len(self.grid_gallery)))
        print("  viper train   | {:8d} | {:8d} "
              .format(len(self.viper_query), len(self.viper_gallery)))
        print("  ilid train    | {:8d} | {:8d} "
              .format(len(self.ilid_query), len(self.ilid_gallery)))
