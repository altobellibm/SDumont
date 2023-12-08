import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from net_model import Net
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch
import torchvision
import os
from torchvision import datasets, transforms


n_epochs = 3
batch_size_train = 64
batch_size_test = 1000
learning_rate = 0.01
momentum = 0.5
log_interval = 10

random_seed = 1
torch.backends.cudnn.enabled = False
torch.manual_seed(random_seed)


#load dataset folder
# Define your transformations
transform = transforms.Compose([
    transforms.Grayscale(),  # Convert to grayscale if your images are in RGB
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# Create datasets
train_dataset = datasets.ImageFolder(root='dataset/train', transform=transform)


# DataLoader instances
train_loader = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True)

#test format
examples = enumerate(train_loader)
batch_idx, (example_data, example_targets) = next(examples)
print(example_data.shape)
os.makedirs('results', exist_ok=True)

# a = 1
network = Net()
optimizer = optim.SGD(network.parameters(), lr=learning_rate,
                      momentum=momentum)

train_losses = []
train_counter = []
test_losses = []
test_counter = [i*len(train_loader.dataset) for i in range(n_epochs + 1)]

def train(epoch):
  network.train()
  for batch_idx, (data, target) in enumerate(train_loader):
    optimizer.zero_grad()
    output = network(data)
    loss = F.nll_loss(output, target, reduction='mean')
    loss.backward()
    optimizer.step()
    if batch_idx % log_interval == 0:
      print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        epoch, batch_idx * len(data), len(train_loader.dataset),
        100. * batch_idx / len(train_loader), loss.item()))
      train_losses.append(loss.item())
      train_counter.append(
        (batch_idx*64) + ((epoch-1)*len(train_loader.dataset)))
      torch.save(network.state_dict(), './results/model.pth')
      torch.save(optimizer.state_dict(), './results/optimizer.pth')

for epoch in range(1, n_epochs + 1):
  train(epoch)