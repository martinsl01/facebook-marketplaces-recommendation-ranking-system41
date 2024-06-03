from pytorch_dataset import ProductsDataset
import torch 
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import time
import copy

class ImageClassifier(torch.nn.Module):

    '''This loads the Resnet50 pre-trained model from torch hub and alters the final layer.'''
    def __init__(self):
        super().__init__()
        self.resnet50 = torch.hub.load(
            'NVIDIA/DeepLearningExamples:torchhub', 'nvidia_resnet50', pretrained=True)
        for param in self.resnet50.parameters(): 
            param.requires_grad = False
        self.added_layer = torch.nn.Sequential( 
            torch.nn.ReLU(),
            torch.nn.Linear(1000, 13))
        self.combination = torch.nn.Sequential( 
            self.resnet50, self.added_layer) 

    def forward(self,X):
        return self.combination(X)

def train(model, train_dataloader, epochs=10):


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
    optimizer = torch.optim.SGD(model.parameters(),lr=0.01)
    writer = SummaryWriter()    
    batch_idx = 0
    print(device)
    model.to(device)
    best_weights = copy.deepcopy(model.state_dict())


    for epoch in range(epochs):
        for i, (features,labels) in enumerate(train_dataloader):
            features = features.to(device)
            print(features.shape)
            labels = labels.to(device)
            prediction = model(features)
            loss = F.cross_entropy(prediction,labels)
            loss.backward()
            print(f"Epoch:{epoch}, Batch number:{i}, Training: Loss: {loss.item()}")
            optimizer.step()
            optimizer.zero_grad()   
            writer.add_scalar('loss',loss.item(),batch_idx)
            batch_idx += 1

dataset = ProductsDataset()
# Split the dataset
train_size = int(0.7 * len(dataset))
val_size = int(0.15 * len(dataset))
test_size = len(dataset) - train_size - val_size
train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=True)
model = ImageClassifier()
train(model, train_dataloader, epochs=10)

torch.save(model.state_dict(), f'C:/Users/marti/FMRRS/model_evaluation/{time.time}')
weight = (model.state_dict())
torch.save(weight['fc.weights'],model.state_dict(), f'C:/Users/marti/FMRRS/model_evaluation/weights/{time.time}')

