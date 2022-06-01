import splitfolders
import torch
import torchvision
from torch import sequential
from torch.nn import conv2d, MaxPool2d, functional
from torch import optim

# data 
train = torchvision.datasets.CIFAR10(root='./data', train=True,download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(train, batch_size=batch_size,shuffle=True, num_workers=2)
# tess
test = torchvision.datasets.CIFAR10(root='/data',test = True, download = True, transform = transform)
test_loader = torch.utils.data.DataLoader(test, batch_size = batch_size, shuffle = True, num_workers = 2)
#classes 
classes = ('plane','car','bird','cat','deer','dog','frog','horse','ship','truck')
                                                     
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
print(model)

optimizer = optim.SGD(lr=0.001)
loss = nn.CrossEntropyLoss()

