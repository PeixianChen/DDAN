from __future__ import print_function, absolute_import
import os.path as osp
import numpy as np
import pdb
from glob import glob
import re
import random


class TestData(object):

    def __init__(self, data_dir):

        self.prid_images_dir = osp.join(data_dir, "DG/PRID")
        self.grid_images_dir = osp.join(data_dir, "DG/GRID")
        self.viper_images_dir = osp.join(data_dir, "DG/VIPeR")
        self.ilid_images_dir = osp.join(data_dir, "DG/iLIDS")

        self.PRID_ID = random.sample([i for i in range(1, 201)], 100)
        self.GRID_ID = random.sample([i for i in range(1, 251)], 125)
        self.VIPeR_ID = random.sample([i for i in range(1, 633)], 316)
        self.ILID_ID = random.sample([i for i in range(1,120)], 60)


        self.PRID_Pair = [i for i in range(1, 201)]
        self.GRID_Pair = [i for i in range(1, 251)]
        self.VIPeR_Pair = [i for i in range(1, 633)]


        # training image dir
        self.gallery_path = 'gallery_test'
        self.query_path = 'query_test'
        self.load()

    def preprocess(self, images_dir, path, queryID, IDpair,  relabel=True):
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
            # if path == self.query_path  and pid not in queryID:
                # continue
            # if path == self.gallery_path and (pid not in queryID and pid in IDpair):
                # continue
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

    def ILIDS_preprocess(self, images_dir, path, queryID, relabel=True):
        domain = 5
        ilid_query, ilid_gallery = [], []
        queryid, galleryid = [], []

        pattern = re.compile(r'([-\d]+)_c?(\d)')
        all_pids = {}
        ret = []
        fpaths = glob(osp.join(images_dir, path, '*.jpg'))
        type = ['*jpeg', '*.png', '*bmp']
        t = 0
        while fpaths == []:
            fpaths = glob(osp.join(images_dir, path, type[t]))
            t += 1
        for fpath in fpaths:
            fname = osp.basename(fpath)
            pid, cam = map(int, pattern.search(fname).groups())
            # if (pid not in queryID) or ((pid in queryid) and (pid in galleryid)):
                # continue
            if pid == -1: continue
            if relabel:
                if pid not in all_pids:
                    all_pids[pid] = len(all_pids)
            else:
                if pid not in all_pids:
                    all_pids[pid] = pid
            pid = all_pids[pid]
            cam -= 1
            if pid not in queryid:
                ilid_query.append((fname, pid, cam, cam))
                queryid.append(pid)
            else:
                ilid_gallery.append((fname, pid, cam,cam))
                galleryid.append(pid)
        return ilid_query, ilid_gallery, int(len(all_pids))

    def load(self):
        # DG TEST
        self.prid_gallery, self.num_gallery_ids = self.preprocess(self.prid_images_dir, self.gallery_path, self.PRID_ID, self.PRID_Pair, False)
        self.prid_query, self.num_query_ids = self.preprocess(self.prid_images_dir, self.query_path, self.PRID_ID, self.PRID_Pair, False)
        self.grid_gallery, self.num_gallery_ids = self.preprocess(self.grid_images_dir, self.gallery_path, self.GRID_ID, self.GRID_Pair, False)
        self.grid_query, self.num_query_ids = self.preprocess(self.grid_images_dir, self.query_path, self.GRID_ID, self.GRID_Pair, False)
        self.viper_gallery, self.num_gallery_ids = self.preprocess(self.viper_images_dir, self.gallery_path, self.VIPeR_ID, self.VIPeR_Pair, False)
        self.viper_query, self.num_query_ids = self.preprocess(self.viper_images_dir, self.query_path, self.VIPeR_ID, self.VIPeR_Pair, False)

        self.ilid_query, self.ilid_gallery, self.num_query_ids = self.ILIDS_preprocess(self.ilid_images_dir, "images", self.ILID_ID, False)

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
