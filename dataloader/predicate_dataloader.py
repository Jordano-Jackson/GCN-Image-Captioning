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
import torchvision.transforms as transforms
import numpy as np
import os
import requests
from io import BytesIO

VG_PATH = "/home/csjihwanh/Desktop/Projects/GCN_Image_Captioning/datasets/vg/"

image_data = json.load(open(os.path.join(VG_PATH, 'image_data.json')))
vg_sgg = h5py.File('/home/csjihwanh/Desktop/Projects/GCN_Image_Captioning/datasets/vg/VG-SGG-with-attri.h5')
vg_sgg_original = h5py.File('/home/csjihwanh/Desktop/Projects/GCN_Image_Captioning/datasets/vg/VG-SGG.h5')
vg_sgg_dicts = json.load(open('/home/csjihwanh/Desktop/Projects/GCN_Image_Captioning/datasets/vg/VG-SGG-dicts-with-attri.json'))

### todo : make dataloader contains bounding box and it's label

class PredicateDataset(Dataset) :
    def __init__(self) :
        print("Loading relationship data..")
        self.rel_num = vg_sgg['relationships'].shape[0]
        self.img_num = len(image_data)
        self.rel_to_img = {}

        for img_idx in range(108072) : # it must be 'range(self.img_num) :' but changed cuz of error
            rth_s = vg_sgg['img_to_first_rel'][img_idx]
            rth_e = vg_sgg['img_to_last_rel'][img_idx]
            if(rth_s == -1) : # when there's no relationship
                continue
            for rth in range(rth_s, rth_e+1) :
                if rth % 100000 == 0 :
                    print(f"{rth} relationships are loaded.")
                self.rel_to_img[rth] = img_idx
        
        print("Loading relationship data done.")
    
    def __len__(self) :
        return self.rel_num

    def bbox_normalize(self, box, height, width) :
        # box : numpy.ndarray
        box[:2] = box[:2]-box[2:]/2
        box[2:] = box[:2]+box[2:]
        box = box.astype(float) / 512 * max(height,width)
        return box.astype(np.float32)

    def __getitem__(self, rel_idx) :
        # return [image, object1 bbox, object2 bbox, predicate]
        transform = transforms.Compose([
            transforms.ToTensor(),
            #transforms.Resize((224,224)),
            transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
        ])
        
        img_idx = self.rel_to_img[rel_idx]

        # load image and transform it into a tensor
        filename = "/home/csjihwanh/Desktop/Projects/GCN_Image_Captioning/datasets/vg/VG_100K/{}.jpg".format(str(image_data[img_idx]['image_id']))
        img = Image.open(filename).convert("RGB")
        if(len(img.getbands()) != 3) :
            img.show()
        img = transform(img)
        height, width = image_data[img_idx]['height'], image_data[img_idx]['width']

        rel = vg_sgg['relationships'][rel_idx]
        predicate = vg_sgg['predicates'][rel_idx][0]

        # fit the box size into the image size
        obj1_bbox = vg_sgg['boxes_512'][rel[0]]
        obj1_bbox = self.bbox_normalize(obj1_bbox, height, width)
        obj2_bbox = vg_sgg['boxes_512'][rel[1]]
        obj2_bbox = self.bbox_normalize(obj2_bbox, height, width) 

        return img, obj1_bbox, obj2_bbox, predicate
        
