import torchvision
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torchvision import transforms
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch.utils.tensorboard import SummaryWriter

from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor
from torchvision.models import resnet50

import matplotlib.pyplot as plt
import numpy as np

from helper_functions import plot_classes_preds


VALIDATION_SIZE = 0.15
BATCH_SIZE = 64
PROGRESS_EVERY = 5
ACCURACY_IN_EPOCH = 5
EPOCHS = 50

writer = SummaryWriter(comment=f'-BATCH_SIZE={BATCH_SIZE}-EPOCHS={EPOCHS}')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(device)

torch.cuda.empty_cache()

classes = ['airplane', 'bird', 'building', 'deer', 'dog', 'frog', 'horse', 'ship', 'sportscar', 'truck']

transform = transforms.Compose(
    [transforms.ToTensor()
     ])

dataset = ImageFolder(root="data/", transform=transform)

print('Dataset loaded')

dataset_size = len(dataset)
indices = list(range(dataset_size))
split = int(np.floor(VALIDATION_SIZE * dataset_size))
np.random.shuffle(indices)
train_indices, val_indices = indices[split:], indices[:split]

train_sampler = SubsetRandomSampler(train_indices)
validation_sampler = SubsetRandomSampler(val_indices)

train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, sampler=train_sampler)
validation_loader = DataLoader(dataset, batch_size=BATCH_SIZE, sampler=validation_sampler)

print('Data loaders initialized')

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.cnn_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.25),

            nn.Conv2d(32, 64, kernel_size=5, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(p=0.25),

            nn.Conv2d(64, 128, kernel_size=3, stride=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.25),

            nn.Conv2d(128, 256, kernel_size=3, stride=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(p=0.25),

            nn.Conv2d(256, 512, kernel_size=3, stride=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(p=0.25),

            nn.Conv2d(512, 1024, kernel_size=3, stride=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(p=0.25)
        )
        
        self.linear_layers = nn.Sequential(
            # nn.Linear(128 * 62 * 62, 500),
            # nn.Linear(64 * 126 * 126, 500),
            nn.Linear(1024 * 14 * 14, 5000),
            nn.BatchNorm1d(5000),
            nn.Dropout(p=0.25),
            nn.Linear(5000, 500),
            nn.BatchNorm1d(500),
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

accuracy_batchess = np.linspace(0, len(train_loader), num=ACCURACY_IN_EPOCH, endpoint=False, dtype=np.int)

for epoch in range(EPOCHS):
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        images, labels = data

        if torch.cuda.is_available():
            images = torch.cuda.FloatTensor(images.numpy())
            labels = torch.cuda.LongTensor(labels.numpy())
        
        # grid = torchvision.utils.make_grid(images)
        # writer.add_image('images', grid, 0)
        # writer.add_graph(model, images)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if i % PROGRESS_EVERY == 0:
            print(f'Epoch: {epoch}, batch: {i}, loss: {running_loss / PROGRESS_EVERY : .3f}')
            # ...log the running loss
            writer.add_scalar('training loss',
                            running_loss / PROGRESS_EVERY,
                            (i + epoch * len(train_loader)))

            running_loss = 0.0

        if i in accuracy_batchess:
            correct = 0
            total = 0
            with torch.no_grad():
                for data in validation_loader:
                    images, labels = data
                    if torch.cuda.is_available():
                        images = torch.cuda.FloatTensor(images.numpy())
                        labels = torch.cuda.LongTensor(labels.numpy())

                    outputs = model(images)
                    _, predicted = torch.max(outputs.cpu().data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels.cpu()).sum().item()

                writer.add_scalar('Accuracy',
                                100 * correct / total,
                                i + epoch * len(train_loader))

                print(f'Accuracy: {100 * correct / total : .3f}%')

        
writer.close()
