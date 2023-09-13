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
    self.resnet101_model = self.resnet101_model.to('cuda' if torch.cuda.is_available() else 'cpu')

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
    return hook

  def detect_semantic_relationships(self, img, v_i, v_j) :
    # vi, vj, vij are list with four elements [x1, y1, x2, y2]
    batch_size = v_i.shape[0]

    # first get three bounding boxes
    v_ij = self.union_bbox(v_i, v_j)

    # get Res4b22 feature map
    y = self.resnet101_model(img.detach())
    output = self.activation['Res4b22'].detach()

    # RoI pooling output = (1024, 7, 7)
    v_i = self.RoIpool(output, [v_i], output_size = (7,7), spatial_scale = 1/16)
    v_j = self.RoIpool(output, [v_j], output_size = (7,7), spatial_scale = 1/16)
    v_ij = self.RoIpool(output, [v_ij], output_size = (7,7), spatial_scale = 1/16)

    # Pool5 pooling output = (2048)
    v_i = self.Pool5(v_i).squeeze()
    v_j = self.Pool5(v_j).squeeze()
    v_ij = self.Pool5(v_ij).squeeze()

    return torch.cat((v_i, v_j, v_ij)) # Output size is 2048*3

  def union_bbox(self, box1, box2):
      x1, y1, x2, y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
      x3, y3, x4, y4 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

      min_x = torch.min(x1, x3)
      min_y = torch.min(y1, y3)
      max_x = torch.max(x2, x4)
      max_y = torch.max(y2, y4)

      return torch.stack([min_x, min_y, max_x, max_y], dim=1)
