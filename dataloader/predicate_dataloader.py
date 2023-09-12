### predicate_dataloader ###
# make dataloader and dataset for predicate_predictor

import h5py
import json
import math
from math import floor
from PIL import Image, ImageDraw
import random

import torch
from torch.utils.data import Dataset, DataLoader 
import numpy as np
import os
import requests
from io import BytesIO

VG_PATH = "/home/csjihwanh/Desktop/Projects/GCN-Image-Captioning/datasets/vg/"

image_data = json.load(open(os.path.join(VG_PATH, 'image_data.json')))
vg_sgg = h5py.File('/home/csjihwanh/Desktop/Projects/GCN-Image-Captioning/datasets/vg/VG-SGG-with-attri.h5')
vg_sgg_original = h5py.File('/home/csjihwanh/Desktop/Projects/GCN-Image-Captioning/datasets/vg/VG-SGG.h5')
vg_sgg_dicts = json.load(open('/home/csjihwanh/Desktop/Projects/GCN-Image-Captioning/datasets/vg/VG-SGG-dicts-with-attri.json'))

### todo : make dataloader contains bounding box and it's label

img_idx = 12321

ith_s = vg_sgg['img_to_first_box'][img_idx]
ith_e = vg_sgg['img_to_last_box'][img_idx]
rth_s = vg_sgg['img_to_first_rel'][img_idx]
rth_e = vg_sgg['img_to_last_rel'][img_idx]
num_objs = ith_e - ith_s + 1
num_rels = rth_e - rth_s + 1


class predicateDataset(Dataset) :
    def __init__(self) :
        print("Loading relationship data..")
        self.rel_num = vg_sgg['relationships'].shape[0]
        self.img_num = len(image_data)
        self.rel_to_img = {}

        for img_idx in range(self.img_num) :
            rth_s = vg_sgg['img_to_first_rel'][img_idx]
            rth_e = vg_sgg['img_to_last_rel'][img_idx]
            for rth in range(rth_s, rth_s) :
                self.rel_to_img[rth] = img_idx
        
        print("Loading relationship data done.")
    
    def __len__(self) :
        return self.rel_num
    
    def __getitem__(self, rel_idx) :
        # return [image index, object1 bbox, object2 bbox, predicate]
        rel = vg_sgg['relationships'][rel_idx]
        predicate = vg_sgg['predicates'][rel_idx]
        obj1_bbox = vg_sgg['boxes_512'][rel[0]]
        obj2_bbox = vg_sgg['boxes_512'][rel[1]]

        return self.rel_to_img[rel_idx], obj1_bbox, obj2_bbox, predicate

