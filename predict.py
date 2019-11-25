import torch
from torchvision import transforms
from PIL import Image

classes = ['airplane', 'bird', 'building', 'deer', 'dog', 'frog', 'horse', 'ship', 'sportscar', 'truck']

model = torch.load('models/resnet50-medium_dataset-with_reduce_lr_on_plateau-scheduler-100epochs')
model.eval()

transform = transforms.ToTensor()

img = Image.open('horse.jpg')
img = img.resize((256, 256), Image.ANTIALIAS)
img = img.convert('RGB')

img = transform(img)
img = img.view(1, 3, 256, 256).cuda()
output = model(img).cpu()

prediction = int(torch.max(output.data, 1)[1].numpy())

print(classes[prediction])
