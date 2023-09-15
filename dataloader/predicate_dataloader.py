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

from config.path_catalog import path_get

image_data = json.load(open(path_get('image_data')))
vg_sgg = h5py.File(path_get('vg_sgg'))
vg_sgg_dicts = json.load(open(path_get('vg_sgg_dicts')))
corrupted_ims = [1592, 1722, 4616, 4617]

### todo : make dataloader contains bounding box and it's label

class PredicateDataset(Dataset) :
    def __init__(self) :
        with open(path_get('rel_to_img'), 'r') as json_file:
            self.rel_to_img = json.load(json_file)
    
    def __len__(self) :
        return vg_sgg['relationships'].shape[0] 

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
        
        img_idx = self.rel_to_img[str(rel_idx)]
        for cor_idx in corrupted_ims :
            if cor_idx <= img_idx :
                img_idx+=1

        # load image and transform it into a tensor
        filename = "/home/csjihwanh/Desktop/Projects/GCN_Image_Captioning/datasets/vg/VG_100K/{}.jpg".format(str(image_data[img_idx]['image_id']))
        img = Image.open(filename).convert("RGB")
        if(len(img.getbands()) != 3) :
            img.show()
        img = transform(img)
        height, width = image_data[img_idx]['height'], image_data[img_idx]['width']

        rel = vg_sgg['relationships'][rel_idx]
        predicate = vg_sgg['predicates'][rel_idx][0]

        # for debugging
        """
        idx_to_label = vg_sgg_dicts['idx_to_label']
        print ('img idx :', img_idx, 'img data', image_data[img_idx])
        print('rel_idx :', rel_idx)
        ith_s = vg_sgg['img_to_first_box'][img_idx]
        ith_e = vg_sgg['img_to_last_box'][img_idx]
        rth_s = vg_sgg['img_to_first_rel'][img_idx]
        rth_e = vg_sgg['img_to_last_rel'][img_idx]
        print('idx range :', ith_s, ith_e, 'rth range :',rth_s, rth_e)
        for obj_idx in range(ith_s, ith_e+1) :
            label = vg_sgg['labels'][obj_idx]
            print(vg_sgg['boxes_1024'][obj_idx])
            print(vg_sgg['boxes_512'][obj_idx])
            print(label, idx_to_label[str(int(label))])
        idx_to_label = vg_sgg_dicts['idx_to_label']
        rel1_label = vg_sgg['labels'][rel[0]]
        rel2_label = vg_sgg['labels'][rel[1]]
        print(rel,rel1_label,rel2_label, idx_to_label[str(int(rel1_label))], idx_to_label[str(int(rel2_label))])
        """

        # fit the box size into the image size
        obj1_bbox = vg_sgg['boxes_512'][rel[0]]
        obj1_bbox = self.bbox_normalize(obj1_bbox, height, width)
        obj2_bbox = vg_sgg['boxes_512'][rel[1]]
        obj2_bbox = self.bbox_normalize(obj2_bbox, height, width) 

        return img, obj1_bbox, obj2_bbox, predicate
        
