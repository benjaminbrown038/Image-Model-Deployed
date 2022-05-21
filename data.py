import splitfolders
# test, train, val
splitfolders.ratio("Images", output="Augmented",seed=1337, ratio=(.8, .1, .1), group_prefix=None, move=False)
#https://pytorch.org/vision/stable/index.html
# data 
import torchvision
train = torchvision.datasets.CIFAR10(root='./data', train=True,download=True, transform=transform)

trainloader = torch.utils.data.DataLoader(train, batch_size=batch_size,shuffle=True, num_workers=2)

test = torchvision.datasets.CIFAR10(root='/data',test = True, download = True, transform = transform)

test_loader = torch.utils.data.DataLoader(test, batch_size = batch_size, shuffle = True, num_workers = 2)

classes = ('plane','car','bird','cat','deer','dog','frog','horse','ship','truck')

# augment
import torchvision.transforms as T
# training augmentation types: horizontal flip, rotation, crop, normalize
# 12 augmentation strategies
train_transform = T.Compose(T.GaussianBlur((3,3)),
                        T.ColorJitter(),
                        T.RandomHorizontalFLip(0.5),
                        T.RandomRotation(0.5),
                        T.RandomVerticalFlip(0.5),
                        T.GaussianBlur((3,3),
                        T.LinearTransformation(),
                        T.adjust_brightness(),
                        T.adjust_contrast(),
                        T.adjust_hue(),
                        T.adjust_saturation(),
                        T.adjust_sharpness())


import torch
from torch import sequential
from torch.nn import conv2d, MaxPool2d
from torch.nn import functional
                            
input_shape = (150,150)
                            
model = nn.Sequential(
            
            nn.Conv2d(3, 32, kernel_size = 3, padding = 1),
            nn.ReLU(),
            nn.Conv2d(32,64, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
        
            nn.Conv2d(64, 128, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(),
            nn.Conv2d(128 ,128, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            
            nn.Conv2d(128, 256, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(),
            nn.Conv2d(256,256, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            
            nn.Flatten(),
            nn.Linear(82944,1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512,10)
        )
