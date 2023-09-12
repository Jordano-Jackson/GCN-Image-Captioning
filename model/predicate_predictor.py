###  predicate predictor ###
# predict predicates between two objects

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader 
import numpy as np

### todo 
# 1. make model 
# model output means predicates, which consists of 50+1 classes
# last class is non-relation class, if prob of non-relation class is higher than 0.5,
# the two object is not connected with relational edge

class PredicatePredictor(nn.Module) :
    def __init__(self) :
        super(PredicatePredictor, self).__init__()
        self.fc1 = nn.Linear(2048*3, 512*3)
        self.fc2 = nn.Linear(512*3, 51)
    
    def forward(self, x) :
        x = F.relu(self.fc(x))
        x = self.fc2(x)
        return x
    
