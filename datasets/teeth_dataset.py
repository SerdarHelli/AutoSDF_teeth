"""
    adopted from: https://github.com/shubhtuls/PixelTransformer/blob/03b65b8612fe583b3e35fc82b446b5503dd7b6bd/data/shapenet.py
"""
import os.path
import json

import h5py
import numpy as np
from PIL import Image
from termcolor import colored, cprint

import torch
import torch.nn.functional as F
import torchvision.utils as vutils
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from torchvision.transforms.functional import InterpolationMode

from datasets.base_dataset import BaseDataset

from utils.util_3d import read_sdf
from utils import binvox_rw

from configs.paths import dataroot


def get_code_setting(opt):
    code_setting = f'{opt.vq_model}-{opt.vq_dset}-{opt.vq_cat}-T{opt.trunc_thres}'
    if opt.vq_note != 'default':
        code_setting = f'{code_setting}-{opt.vq_note}'
    return code_setting

    
class TeethDataset(BaseDataset):

    def initialize(self, opt, phase='train', cat='all'):
        self.opt = opt
        self.max_dataset_size = opt.max_dataset_size

        self.path=opt.vq_data_path
        self.files=os.listdir(self.path)
        self.N = len(self.files)
        self.trunc_tresh=0.2


    def __getitem__(self, index):

        # model_id = self.model_list[index]
        # sdf_h5_file = osp.join(self._data_dir, 'SDF_v1', synset, model_id, 'ori_sample_grid.h5')
        path_sdf = self.files[index]
        
        sdf = np.load(os.path.join(self.path,path_sdf))
        sdf = torch.Tensor(sdf).view(1, 64, 64, 64)
        # print(sdf.shape)
        # sdf = sdf[:, :64, :64, :64]

        thres = self.opt.trunc_thres
        if thres != 0.0:
            sdf = torch.clamp(sdf, min=-thres, max=thres)

        ret = {
            'sdf': sdf,
            'cat_id': 0,
            'cat_str': "teeth",
            'path': path_sdf,
            # 'tsdf': tsdf,
        }

        return ret

    def __len__(self):
        return self.N

    def name(self):
        return 'TeethDataset'
    
