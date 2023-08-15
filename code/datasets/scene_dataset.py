import os
import torch
import numpy as np

import utils.general as utils
from utils import rend_util

class SceneDataset(torch.utils.data.Dataset):

    def __init__(self,
                scan_id=0,
                cam_file=None
                ):
        ''' 源码的训练规模：49 张照片，每张照片随机选 2048 个 pixel'''
        
        # ../data/ys9
        self.instance_dir = os.path.join('../data', 'ys{0}'.format(scan_id))
        assert os.path.exists(self.instance_dir), "Data directory is empty"
        points_dir = '{0}/points'.format(self.instance_dir)
        points_paths = sorted(utils.glob_numpy(points_dir))
        pose_dir = '{0}/pose'.format(self.instance_dir)
        pose_paths = sorted(utils.globa_numpy(pose_dir))
        # 8
        self.n_images = len(image_paths)

        # 读取 pose
        self.pose_all = []
        for path in pose_paths:
            self.pose_all.append(torch.from_numpy(np.load(path)).float())
        # 读取 points
        self.points_all = []
        for path in points_paths:
            self.points_all.append(torch.from_numpy(np.load(path)).float())

    def __len__(self):
        # 8
        return self.n_images

    def __getitem__(self, idx):

        sample = {
            "pose": self.pose_all[idx],
            "points": self.points_all[idx]
        }

        return idx, sample

    def collate_fn(self, batch_list):
        batch_list = zip(*batch_list)
        all_parsed = []
        # batch_size = 1
        for entry in batch_list:
            if type(entry[0]) is dict:
                ret = {}
                for k in entry[0].keys():
                    ret[k] = torch.stack([obj[k] for obj in entry])
                all_parsed.append(ret)
            else:
                all_parsed.append(torch.LongTensor(entry))
        return tuple(all_parsed)

