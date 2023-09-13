import numpy as np

import torch 
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision

from model.predicate_predictor import PredicatePredictor
from dataloader.predicate_dataloader import PredicateDataset
from utils.bbox_feature_extractor import BboxFeatureExtractor

def predicate_train(model, train_loader, num_epochs, learning_rate) :
    if torch.cuda.is_available() :
        print("cuda is available.")
        device = torch.device("cuda")
    else :
        print("cuda is not available.")
        device = torch.device("cpu")
    
    model = model.to('cuda' if torch.cuda.is_available() else 'cpu')
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr = learning_rate)
    bbox_extractor = BboxFeatureExtractor()
    
    print("Start training...")
    for epoch in range(num_epochs) :
        total_loss = 0.0
        model.train()

        for img, bbox1, bbox2, predicate in train_loader :
            one_hot_vector = torch.zeros(51)
            one_hot_vector[predicate] = 1
            predicate = one_hot_vector

            img, bbox1, bbox2, predicate = img.to(device), bbox1.to(device), bbox2.to(device), predicate.to(device)

            # gradient initialization
            optimizer.zero_grad()

            x = bbox_extractor.detect_semantic_relationships(img, bbox1, bbox2)

            outputs = model(x)
            
            loss = criterion(outputs, predicate)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch [{epoch + 1}/{num_epochs}] - Loss : {total_loss / len(train_loader)}")
            
    print("Training finished.")

dataset = PredicateDataset()

batch_size = 1
epochs = 10
lr= 1e-4
dataloader = DataLoader(dataset, batch_size =batch_size, shuffle=True)

predicate_model = PredicatePredictor()

predicate_train(predicate_model, dataloader, num_epochs= 10 , learning_rate=lr)