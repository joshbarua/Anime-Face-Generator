import torch
import argparse
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from torch.utils.data.dataloader import DataLoader
from torchvision.utils import save_image
from torchvision.utils import make_grid
from torch.utils.tensorboard import SummaryWriter 

#loading and preprocessing data

#resize images to fixed size, convert image to tensor, normalize pixel image from [0, 255] to [-1, 1]
train_transform = transforms.Compose([transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize([.5, .5, .5], [.5, .5, .5])])

#import data and apply transformations
train_data = datasets.ImageFolder(root='faces', transform=train_transform)

#create iterable over faces dataset
train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)


