#import numpy as np
#import torch
#from pathlib import Path
#import models

from __future__ import division

import os, sys, time, random, argparse, PIL
from pathlib import Path
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True # please use Pillow 4.0.0 or it may fail for some images
from os import path as osp
import numbers, numpy as np
import init_path
import torch
import models
import datasets
from visualization import draw_image_by_points
from san_vision import transforms
from utils import time_string, time_for_file, get_model_infos


if torch.cuda.is_available():
  device = 'cuda'
else:
  device = 'cpu'


class LandmarkDetector:
  def __init__(self, model_path):
    snapshot = Path(model_path)
    if device == 'cpu':
        snapshot = torch.load(snapshot, map_location='cpu')
    else:
        snapshot = torch.load(snapshot)

    mean_fill   = tuple( [int(x*255) for x in [0.5, 0.5, 0.5] ] )
    normalize   = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    self.param = snapshot['args']
    eval_transform  = transforms.Compose([transforms.PreCrop(self.param.pre_crop_expand), transforms.TrainScale2WH((self.param.crop_width, self.param.crop_height)),  transforms.ToTensor(), normalize])

    self.net = models.__dict__[self.param.arch](self.param.modelconfig, None)

    if device == 'cuda':
        self.net = self.net.cuda()

    weights = models.remove_module_dict(snapshot['state_dict'])
    self.net.load_state_dict(weights)

    self.dataset = datasets.GeneralDataset(eval_transform, self.param.sigma, self.param.downsample, self.param.heatmap_type, self.param.dataset_name)
    self.dataset.reset(self.param.num_pts)


  def preprocess_image(self, image_path, face):
    [image, _, _, _, _, _, self.cropped_size], meta = self.dataset.prepare_input(image_path, face)    

    with torch.no_grad():
        if device == 'cpu':
            self.inputs = image.unsqueeze(0)
        else:
            self.inputs = image.unsqueeze(0).cuda()

  def predict(self):
    error_message = ''
    with torch.no_grad():

        batch_heatmaps, batch_locs, batch_scos, _ = self.net(self.inputs)
    
    cpu = torch.device('cpu')
    np_batch_locs, np_batch_scos, cropped_size = batch_locs.to(cpu).numpy(), batch_scos.to(cpu).numpy(), self.cropped_size.numpy()
    locations, scores = np_batch_locs[0,:-1,:], np.expand_dims(np_batch_scos[0,:-1], -1)

    scale_h, scale_w = cropped_size[0] * 1. / self.inputs.size(-2) , cropped_size[1] * 1. / self.inputs.size(-1)

    locations[:, 0], locations[:, 1] = locations[:, 0] * scale_w + cropped_size[2], locations[:, 1] * scale_h + cropped_size[3]
    self.prediction = np.concatenate((locations, scores), axis=1).transpose(1,0)

    landmarks = {}
    for i in range(self.param.num_pts):
        point = self.prediction[:, i]
        landmarks[(float(point[0]), float(point[1]))] = float(point[2])

    return landmarks, error_message

