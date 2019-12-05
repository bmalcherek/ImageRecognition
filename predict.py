import torch
from torchvision import transforms
from torchvision.models import resnet50

from PIL import Image
import os

classes = ['airplane', 'bird', 'building', 'deer', 'dog', 'frog', 'horse', 'ship', 'sportscar', 'truck']

model = resnet50()
model.fc = torch.nn.Linear(2048, 10, bias=True)
model.load_state_dict(torch.load('final_state_dict', map_location=torch.device('cpu')))
model.eval()

transform = transforms.ToTensor()

for image in os.listdir('test_images'):
    try:
        img = Image.open(f'test_images/{image}')
        img = img.resize((256, 256), Image.ANTIALIAS)
        img = img.convert('RGB')

        img = transform(img)
        img = img.view(1, 3, 256, 256)
        output = model(img)

        prediction = int(torch.max(output.data, 1)[1].numpy())

        print(f'Image {image}, predicted class: {classes[prediction]}')
    except:
        print(f'{image} failed')
