import torchvision
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torchvision import transforms
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor
from torchvision.models import resnet50

import matplotlib.pyplot as plt
import numpy as np


VALIDATION_SIZE = 0.15
BATCH_SIZE = 16

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(device)

# torch.cuda.empty_cache()

print(device)

classes = ['airplane', 'bird', 'building', 'deer', 'dog', 'frog', 'horse', 'ship', 'sportscar', 'truck']

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

dataset = ImageFolder(root="data/", transform=transform)
print(type(dataset))

print('Dataset loaded')

dataset_size = len(dataset)
indices = list(range(dataset_size))
split = int(np.floor(VALIDATION_SIZE * dataset_size))
np.random.shuffle(indices)
train_indices, val_indices = indices[split:], indices[:split]

train_sampler = SubsetRandomSampler(train_indices)
validation_sampler = SubsetRandomSampler(val_indices)

train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, sampler=train_sampler, num_workers=2)
validation_loader = DataLoader(dataset, batch_size=BATCH_SIZE, sampler=validation_sampler)

print('Data loaders initialized')

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.cnn_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.25),
            # nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 64, kernel_size=3, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(p=0.25),

            nn.Conv2d(64, 128, kernel_size=3, stride=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(p=0.25)
        )
        
        self.linear_layers = nn.Sequential(
            nn.Linear(128 * 62 * 62, 500),
            nn.Dropout(p=0.25),
            nn.Linear(500, 10)
        )
    
    def forward(self, x):
        x = self.cnn_layers(x)
        # print(x.size())
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)
        return x

model = Net()
# model = resnet50(False)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.003)
print(model)

if torch.cuda.is_available():
    print('Sending model to CUDA')
    model = model.cuda()
    criterion = criterion.cuda()

print('Net created')

for epoch in range(10):
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data

        if torch.cuda.is_available():
            inputs = torch.cuda.FloatTensor(inputs.numpy())
            labels = torch.cuda.LongTensor(labels.numpy())

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if i % 10 == 0:
            print(f'Epoch: {epoch}, batch: {i}, loss: {running_loss / 100 : .3f}')
            running_loss = 0.0

    correct = 0
    total = 0
    with torch.no_grad():
        for data in validation_loader:
            images, labels = data
            if torch.cuda.is_available():
                # print(inputs.device)
                images = torch.cuda.FloatTensor(images.numpy())
                labels = torch.cuda.LongTensor(labels.numpy())

            outputs = model(images)
            _, predicted = torch.max(outputs.cpu().data, 1)
            total += labels.size(0)
            correct += (predicted == labels.cpu()).sum().item()

    print(f'Accuracy: {100 * correct / total : .3f}%')
