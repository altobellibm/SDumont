import torch
import torchvision
import os
from PIL import Image
import torchvision.transforms as transforms

n_epochs = 3
batch_size_train = 64
batch_size_test = 1000
learning_rate = 0.01
momentum = 0.5
log_interval = 10

random_seed = 1
torch.backends.cudnn.enabled = False
torch.manual_seed(random_seed)

def savable_dataset(dataset, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    to_pil = transforms.ToPILImage()

    for i, (data, target) in enumerate(dataset):
        for j, img in enumerate(data):
            label_dir = os.path.join(save_dir, str(target[j].item()))
            os.makedirs(label_dir, exist_ok=True)

            img_path = os.path.join(label_dir, f'img_{i}_{j}.png')
            img = to_pil(img)
            img.save(img_path)


train_loader = torch.utils.data.DataLoader(
  torchvision.datasets.MNIST('/files/', train=True, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])),
  batch_size=batch_size_train, shuffle=True)

test_loader = torch.utils.data.DataLoader(
  torchvision.datasets.MNIST('/files/', train=False, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])),
  batch_size=batch_size_test, shuffle=True)

# save the dataset in folder images
savable_dataset(train_loader, 'dataset/train')
savable_dataset(test_loader, 'dataset/test')