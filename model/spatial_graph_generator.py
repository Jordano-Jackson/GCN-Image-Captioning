### spatial relationship calculator ###
# calculate the type of spatial relationship between two objects 

# todo : when a relationship is given, calcuate the spatial relationship type
# According to 
# Yao, T. (2018, September 19). Exploring visual relationship for image captioning. arXiv.org. https://arxiv.org/abs/1809.07041
# There are 11 types of spatial realtionship
# C1 : Inside
# C2 : Cover
# C3 : Overlap
# C4-11 : Index 

import math
import json 
import os 
import h5py

from config.path_catalog import path_get

class SpatialGraphGenerator() :
    def __init__(self) :

        self.image_data = json.load(open(path_get('image_data')))
        self.vg_sgg = h5py.File(path_get('vg_sgg'))
        self.vg_sgg_dicts = json.load(open(path_get('vg_sgg_dicts')))

    def calc_iou(self, bbox1, bbox2) :
        # calculate intersection over Union

        # calculate intersection rectangle
        x1_intersection = max((bbox1[0], bbox2[0]))
        y1_intersection = max((bbox1[1], bbox2[1]))
        x2_intersection = min((bbox1[2], bbox2[2]))
        y2_intersection = min((bbox1[3], bbox2[3]))

        # calculate the area of intersection rectangle 
        intersection_area = max(0, x2_intersection-x1_intersection) * max(0, y2_intersection-y1_intersection)

        # calculate the areas of individual bounding boxes
        area_bbox1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area_bbox2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])

        # Calculate IoU
        iou = intersection_area / float(area_bbox1 + area_bbox2 - intersection_area)
        return iou

    def inside_or_cover(self, bbox1, bbox2) :
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2

        if x1_1 >= x1_2 and y1_1 >= y1_2 and x2_1 <= x2_2 and y2_1 <= y2_2 :
            return 1 # Inside class
        elif x1_1 <= x1_2 and y1_1 <= y1_2 and x2_1 >= x2_2 and y2_1 >= y2_2 :
            return 2 # Cover class
        else :
            return 0 # Other class

    def calc_angle(self, bbox1, bbox2) :
        center1_x = (bbox1[0] + bbox1[2]) / 2
        center1_y = (bbox1[1] + bbox1[3]) / 2
        center2_x = (bbox2[0] + bbox2[2]) / 2
        center2_y = (bbox2[1] + bbox2[3]) / 2

        # Calculate the vector 
        dx = center2_x - center1_x
        dy = center2_y - center1_y

        angle = math.atan2(dy, dx) # radian
        angle = math.degrees(angle) 

        if angle < 0: 
            angle += 360
        return angle

    def bbox_convert(self, box) :
        # convert (x,y,w,h) to (x1,y1,x2,y2)
        box[:2] = box[:2] - box[2:] / 2
        box[2:] = box[:2] + box[2:]
        return box

    def generate_spatial_graph(self, img_idx) :
        obj_s = self.vg_sgg['img_to_first_box'][img_idx]
        obj_e = self.vg_sgg['img_to_last_box'][img_idx]
        
        vertex = []
        edge = []

        for obj1 in range(obj_s, obj_e + 1) :
            vertex.append(obj1)
            for obj2 in range(obj_s, obj_e + 1) :
                if obj1 == obj2 : continue
                
                bbox1 = self.vg_sgg['boxes_512'][obj1]
                bbox2 = self.vg_sgg['boxes_512'][obj2]
                bbox1 = self.bbox_convert(bbox1)
                bbox2 = self.bbox_convert(bbox2)

                inside_cover_check = self.inside_or_cover(bbox1, bbox2)
                iou = self.calc_iou(bbox1, bbox2)
                angle = self.calc_angle(bbox1, bbox2)

                if inside_cover_check == 1 :
                    edge_class = 1 # Inside class
                elif inside_cover_check == 2 :
                    edge_class = 2 # Cover class
                elif iou >= 0.5 : 
                    edge_class = 3 # Overlap
                else :
                    print(angle)
                    edge_class = math.ceil(angle/45) + 3 # from 4 to 11
                
                edge.append([obj1, obj2, edge_class])

        return vertex, edge

img_idx = 2
sp = SpatialGraphGenerator()
vertex, edge = sp.generate_spatial_graph(img_idx)
print(vertex, edge)