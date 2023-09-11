import h5py
import json
import math
from math import floor
from PIL import Image, ImageDraw
import random

import torch
import numpy as np
import os
import requests
from io import BytesIO

VG_PATH = "/home/csjihwanh/Desktop/Projects/GCN-Image-Captioning/datasets/vg/"

image_data = json.load(open(os.path.join(VG_PATH, 'image_data.json')))
vg_sgg = h5py.File('/home/csjihwanh/Desktop/Projects/GCN-Image-Captioning/datasets/vg/VG-SGG-with-attri.h5')
vg_sgg_original = h5py.File('/home/csjihwanh/Desktop/Projects/GCN-Image-Captioning/datasets/vg/VG-SGG.h5')
vg_sgg_dicts = json.load(open('/home/csjihwanh/Desktop/Projects/GCN-Image-Captioning/datasets/vg/VG-SGG-dicts-with-attri.json'))

USE_BOX_SIZE = 1024

def draw_single_box(pic, box, color = (255,0,255,128)) :
    draw = ImageDraw.Draw(pic)
    x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
    draw.rectangle(((x1,y1), (x2, y2)), outline = color)

def draw_boxes(image_id, boxes) :
    pic = Image.open("/home/csjihwanh/Desktop/Projects/GCN-Image-Captioning/datasets/vg/VG_100K/{}.jpg".format(image_id))
    num_obj = boxes.shape[0] 
    for i in range(num_obj) :
        draw_single_box(pic, boxes[i])
    return pic

def show_box_attributes(image_data, vg_sgg, obj_attributes, vg_sgg_dicts, img_idx = None) :
    idx_to_label = vg_sgg_dicts['idx_to_label']
    idx_to_attribute = vg_sgg_dicts['idx_to_attribute']
    if img_idx is None :
        img_idx = random.randint(0, len(image_data)-1)
    height, width = image_data[img_idx]['height'], image_data[img_idx]['width']
    filename = "/home/csjihwanh/Desktop/Projects/GCN-Image-Captioning/datasets/vg/VG_100K/{}.jpg".format(str(image_data[img_idx]['image_id']))
    pic = Image.open(filename)
    ith_s = vg_sgg['img_to_first_box'][img_idx]
    ith_e = vg_sgg['img_to_last_box'][img_idx]
    obj_idx = random.randint(ith_s, ith_e)
    box = vg_sgg['boxes_1024'][obj_idx]
    label = vg_sgg['labels'][obj_idx]
    attribute = obj_attributes[obj_idx]
    box[:2] = box[:2]-box[2:]/2
    box[2:] = box[:2]+box[2:]
    box = box.astype(float) / USE_BOX_SIZE * max(height,width)
    draw_single_box(pic, box)
    att_list = []
    if attribute.sum() > 0 :
        for i in attribute.tolist():
            if i>0:
                att_list.append(idx_to_attribute[str(i)])
        print('Index: {}, Path : {}'.format(img_idx, filename))
        print('Label: {}'.format(idx_to_label[str(int(label))]))
        print('Attribute: {}'.format(','.join(att_list)))
        return pic
    else :
        return show_box_attributes(image_data, vg_sgg, obj_attributes, vg_sgg_dicts)
    
#show_box_attributes(image_info, vg_sgg, obj_attributes, vg_sgg_dicts)

# todo : retrieve scene graph from the image 
# 1. get all the list of objects per image

def get_scene_graph(vg_sgg, img_idx) :
    idx_to_label = vg_sgg_dicts['idx_to_label']
    idx_to_attribute = vg_sgg_dicts['idx_to_attribute']
    
    ith_s = vg_sgg['img_to_first_box'][img_idx]
    ith_e = vg_sgg['img_to_last_box'][img_idx]
    rth_s = vg_sgg['img_to_first_rel'][img_idx]
    rth_e = vg_sgg['img_to_last_rel'][img_idx]
    num_objs = ith_e - ith_s + 1
    num_rels = rth_e - rth_s + 1
    image_path = image_data[img_idx]['url']
    filename = "/home/csjihwanh/Desktop/Projects/GCN-Image-Captioning/datasets/vg/VG_100K/{}.jpg".format(str(image_data[img_idx]['image_id']))
    img = Image.open(filename)
    img.show()
    #print(num_objs, num_rels)

get_scene_graph(vg_sgg, 1)
    
#num_objs = idx_to_label[str(vg_sgg['labels'][1][0])]

# 2. make dataloader for object classification 

# 3. make dataloader for semantic relation prediction 

