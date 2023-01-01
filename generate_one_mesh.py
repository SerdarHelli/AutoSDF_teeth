import os
import numpy as np
import imageio
from einops import rearrange
from termcolor import colored, cprint
from tqdm import tqdm
from omegaconf import OmegaConf

import torch
import torch.backends.cudnn as cudnn
import torchvision.utils as vutils
cudnn.benchmark = True

from options.test_options import TestOptions
from datasets.dataloader import CreateDataLoader, get_data_generator
from models.base_model import create_model

from configs.paths import dataroot
from utils import util
from utils.util_3d import sdf_to_mesh_savemesh


opt = TestOptions().parse()
opt.phase = 'test'

seed = opt.seed
util.seed_everything(seed)

sdf_file_path=opt.sdf_path
sdf = np.load(sdf_file_path)
sdf = torch.Tensor(sdf).view(1,1, 64, 64, 64)
# print(sdf.shape)
# sdf = sdf[:, :64, :64, :64]

thres = 0.2
if thres != 0.0:
    sdf = torch.clamp(sdf, min=-thres, max=thres)

data = {
    'sdf': sdf,
    'cat_id': 0,
    'cat_str': "teeth",
    'path': sdf_file_path,
    # 'tsdf': tsdf,
}
# main loop
model = create_model(opt)
cprint(f'[*] "{opt.model}" initialized.', 'cyan')

# load ckpt
model.load_ckpt(opt.ckpt)


model.inference(data)
        
mesh=sdf_to_mesh_savemesh(model.x_recon,opt.mesh_path)

