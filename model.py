import torch
import torch.nn as nn
import torch.nn.Functional as F
import torchvision.transforms as T 
from torchvision.utils import make_grid
from torchvision.models import resnet50
from fastai.vision import *
# bring in augmented data 

# testing model against data 
model = resnet50(pretrained=True)


# modifying head 
model = resnet50(pretrained = True)

# Modifying Head - classifier

model.fc = nn.Sequential(
  nn.Linear(2048,1,bias=True),
  nn.Sigmoid()
)

opt = F.optimizer.Adam(lr=.0005,Adam,
                       
                      )

