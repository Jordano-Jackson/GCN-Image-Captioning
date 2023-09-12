import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models.detection
import torchvision.models as models
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights, FasterRCNN_ResNet50_FPN_V2_Weights
from torchvision.io import read_image

class BboxFeatureExtractor :

  def __init__(self) :
    # get feature vector from Res4b22 and RoI pooling layer from ResNet-101
    print("BBoxFeatureExtractor initializing... ", end='')
    self.resnet101_model = models.resnet101(weights="ResNet101_Weights.IMAGENET1K_V2")

    self.activation = {}
    self.resnet101_model.layer3[22].conv3.register_forward_hook(self.get_activation('Res4b22'))
    self.RoIpool = torchvision.ops.roi_pool
    self.Pool5 = nn.Sequential (
        self.resnet101_model.layer4,
        nn.AdaptiveAvgPool2d(output_size =(1,1)),
    )
    print("Done. ")

  def get_activation(self, name) :
    def hook(model, input, output) :
      self.activation[name] = output.detach()
      #print(self.activation[name])
    return hook

  def detect_semantic_relationships(self, img, v_i, v_j) :
    # vi, vj, vij are list with four elements [x1, y1, x2, y2]

    # first get three bounding boxes
    v_ij = self.union_bbox(v_i, v_j)

    # get Res4b22 feature map
    y = self.resnet101_model(img.unsqueeze(0)).detach()
    output = self.activation['Res4b22'].detach()

    # RoI pooling with 3 bounding boxes
    v_i = torch.cat((torch.zeros(1), v_i));
    v_j = torch.cat((torch.zeros(1), v_j));
    v_ij = torch.cat((torch.zeros(1), v_ij));
    v_stack = torch.stack((v_i, v_j, v_ij))

    # RoI pooling output = (1024, 7, 7)
    v_i, v_j, v_ij = self.RoIpool(output, v_stack, output_size = (7,7), spatial_scale = 1/16)

    # Pool5 pooling output = (2048)
    v_i = self.Pool5(v_i.unsqueeze(0)).squeeze()
    v_j = self.Pool5(v_j.unsqueeze(0)).squeeze()
    v_ij = self.Pool5(v_ij.unsqueeze(0)).squeeze()

    return torch.cat((v_i, v_j, v_ij)) # Output size is 2048*3

  def union_bbox(self, box1, box2) :
    x11, y11, x12, y12 = box1;
    x21, y21, x22, y22 = box2;
    return torch.tensor([min(x11,x21), min(y11, y21), min(x12,x22), min(y12,y22)])
