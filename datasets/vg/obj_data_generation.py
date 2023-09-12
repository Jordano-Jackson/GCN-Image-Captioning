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

obj_data = []

for img_idx in range(108072):
    if(img_idx% 10807== 0) :
        print(f"{int(img_idx/10807)}0% data has been processed..");
    ith_s = vg_sgg['img_to_first_box'][img_idx]
    ith_e = vg_sgg['img_to_last_box'][img_idx]
    for obj_idx in range(ith_s, ith_e) :
        bbox = vg_sgg['boxes_512'][obj_idx]
        label = vg_sgg['labels'][obj_idx]
        obj_data.append({
            'bbox': bbox.tolist(),
            'label': label.item(),
            'img_idx': img_idx
        })


print(type(obj_data))

with open('obj_data.json', 'w') as json_file:
    json.dump(obj_data, json_file, indent =4) 

print("obj_data.json is saved")