# -*- coding: utf-8 -*-

import torch
import torchvision
import torchvision.transforms as transforms
from matplotlib import pyplot as plt
from PIL import Image
import numpy as np
import torch.utils.data as data
import os
import os.path

# plt.figure()
# plt.imshow(data, interpolation='nearest')
# plt.show()
# plt.imsave('img3.png',data)

# plt.figure()
# da =  Image.open("Figure_1.png")

dtype = torch.float
device = torch.device("cpu")
# device = torch.device("cuda:0")

N=100
h=70
w=70
Din =(70,70)

class ModelNetDataset(data.Dataset):
    def __init__(self):
        pass
    def __len__(self):
        pass
    
    def __getitem__(self, index):
        pass
    
class SiamNetPoint2D(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
class SiamNetConv(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = torch.nn.Conv2d(1,100, 3, stride=2)
        self.relu  = torch.nn.ReLU()
        self.linear1 = torch.nn.Linear(100*34*34,100)
    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        # print(x.shape)
        x = self.linear1(torch.flatten(x))
        x = self.relu(x)
        
        return x

model = SiamNetConv()

criterion = torch.nn.MSELoss(reduction='sum')
optimizer = torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.9)

img =  Image.open("img1.png").convert('P')
# plt.figure()
# plt.imshow(data, interpolation='nearest')
# plt.show()

img = np.asarray(img, dtype=np.float32)

img = np.expand_dims(img, axis=2)

# if img.ndim == 2:
#     # reshape (H, W) -> (1, H, W)
#     img = img[np.newaxis]

transform = transforms.Compose(
    [transforms.ToTensor(),
     # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
     ])

img = transform(img)
img = img.unsqueeze(0)
y_pred = model(img)

# Compute and print loss
loss = criterion(y_pred, img)

optimizer.zero_grad()
loss.backward()
optimizer.step()


x = torch.randn(N,Din, device=device,dtype=dtype)
y = torch.randn(N,Dout, device=device,dtype=dtype)

w1 = torch.randn(Din,H, device=device,dtype=dtype,requires_grad=True)
w2 = torch.randn(H,Dout, device=device,dtype=dtype,requires_grad=True)

learning_rate = 1e-6
for t in range(500):
    y_pred = x.mm(w1).clamp(min=0).mm(w2)
    loss = (y_pred-y).pow(2).sum()
    if t % 100 == 99:
        print(t, loss.item())
        
    loss.backward()
    
    
    with torch.no_grad():
        w1 -= learning_rate*w1.grad()
        w2 -= learning_rate*w2.grad()
    
        w1.grad.zero_()
        w2.grad.zero_()

            



