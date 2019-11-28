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

# from datetime import datetime

from helper_functions import plot_classes_preds


INPUT_SIZE = 256
PROGRESS_EVERY = 50
ACCURACY_IN_EPOCH = 4
MODEL_NAME = f'resnet50-big_dataset-with_reduce_lr_on_plateau-scheduler-150epochs'

VALIDATION_SIZE = 0.15
BATCH_SIZE = 32
EPOCHS = 150
FIRST_LR = 0.0023

writer = SummaryWriter(comment=f'-BATCH_SIZE={BATCH_SIZE}-EPOCHS={EPOCHS}-LR={FIRST_LR}')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(device)

torch.cuda.empty_cache()

classes = ['airplane', 'bird', 'building', 'deer', 'dog', 'frog', 'horse', 'ship', 'sportscar', 'truck']

transform = transforms.Compose([
        transforms.RandomRotation(25),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])

dataset = ImageFolder(root="big_dataset/", transform=transform)
# dataset = ImageFolder(root="data/", transform=transform)
# dataset = ImageFolder(root="fast_images/", transform=transform)

print(f'Dataset loaded: {len(dataset)} photos')

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

# model = Net()
model = resnet50(False)
model.fc = nn.Linear(2048, 10, bias=True)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=FIRST_LR)
# scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [10, 35])
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.2)
print(model)

if torch.cuda.is_available():
    print('Sending model to CUDA')
    model = model.cuda()
    criterion = criterion.cuda()

print('Net created')

accuracy_batchess = np.linspace(0, len(train_loader) - 1, num=ACCURACY_IN_EPOCH + 1, dtype=np.int)[1:]

# Switch to check validation loss in scheduler and check validation data on last batch instead of first

best_accuracy = 0.0
best_loss = 500000
for epoch in range(EPOCHS):
    epoch_loss = 0.0
    running_loss = 0.0
    running_corrects = 0.0
    for i, data in enumerate(train_loader, 0):
        images, labels = data

        if torch.cuda.is_available():
            images = torch.cuda.FloatTensor(images.numpy())
            labels = torch.cuda.LongTensor(labels.numpy())
        
        # grid = torchvision.utils.make_grid(images)
        # writer.add_image('images', grid, 0)
        # writer.add_graph(model, images)

        # exit()

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        _, predicted = torch.max(outputs.cpu().data, 1)

        running_loss += loss.item()
        epoch_loss += loss.item()
        running_corrects += torch.sum(labels.cpu().data == predicted)

        if i % PROGRESS_EVERY == PROGRESS_EVERY - 1:
            print(f'Epoch: {epoch}, batch: {i + 1}, loss: {running_loss / PROGRESS_EVERY : .3f}, ' +
             f'accuracy: {100 * running_corrects / (BATCH_SIZE * PROGRESS_EVERY)}%')
            # ...log the running loss
            writer.add_scalar('Loss/train',
                            running_loss / PROGRESS_EVERY,
                            i * BATCH_SIZE + epoch * len(train_loader.dataset))
            
            writer.add_scalar('Accuracy/train',
                            100 * running_corrects / (BATCH_SIZE * PROGRESS_EVERY),
                            i * BATCH_SIZE + epoch * len(train_loader.dataset)) 

            running_loss = 0.0
            running_corrects = 0.0

        if i in accuracy_batchess:
            correct = 0
            loss = 0.0
            total = 0
            with torch.no_grad():
                for data in validation_loader:
                    images, labels = data
                    if torch.cuda.is_available():
                        images = torch.cuda.FloatTensor(images.numpy())
                        labels = torch.cuda.LongTensor(labels.numpy())

                    outputs = model(images)
                    _, predicted = torch.max(outputs.cpu().data, 1)

                    loss += criterion(outputs, labels).item()
                    total += labels.size(0)
                    correct += (predicted == labels.cpu()).sum().item()

                writer.add_scalar('Accuracy/test',
                                100 * correct / total,
                                i * BATCH_SIZE + epoch * len(validation_loader.dataset))

                writer.add_scalar('Loss/test',
                                loss / len(validation_loader),
                                i * BATCH_SIZE + epoch * len(validation_loader.dataset))

                print(f'TEST Loss: {loss / len(validation_loader)}, Accuracy: {100 * correct / total : .3f}%')

            if i == accuracy_batchess[-1]:
                loss = loss / len(validation_loader)
                acc = 100 * correct / total
                # if acc > best_accuracy:
                if loss < best_loss:
                    best_loss = loss
                    # torch.save(model, 'models/' + MODEL_NAME)
                    torch.save(model.state_dict(), 'models/' + MODEL_NAME + '-state_dict')
                    print(f'Saving best yet model - {acc:.3f}% accuracy - {loss:.3f} loss')

    epoch_loss = epoch_loss / len(train_loader)
    scheduler.step(epoch_loss)
        
writer.close()
