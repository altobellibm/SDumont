from net_model import Net
import torch.optim as optim
import torch.nn.functional as F
import torch 
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
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

network = Net()
optimizer = optim.SGD(network.parameters(), lr=learning_rate,
                      momentum=momentum)

network.load_state_dict(torch.load('./results/model.pth'))
optimizer.load_state_dict(torch.load('./results/optimizer.pth'))

def test(test_loader):
  test_losses = []
  network.eval()
  test_loss = 0
  correct = 0
  with torch.no_grad():
    for data, target in test_loader:
      output = network(data)
      test_loss += F.nll_loss(output, target, size_average=False).item()
      pred = output.data.max(1, keepdim=True)[1]
      correct += pred.eq(target.data.view_as(pred)).sum()
  test_loss /= len(test_loader.dataset)
  test_losses.append(test_loss)
  print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    test_loss, correct, len(test_loader.dataset),
    100. * correct / len(test_loader.dataset)))
  
#load dataset folder
# Define your transformations
transform = transforms.Compose([
    transforms.Grayscale(),  # Convert to grayscale if your images are in RGB
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# Create datasets
test_dataset = datasets.ImageFolder(root='dataset/test', transform=transform)
test_loader = DataLoader(test_dataset, batch_size=batch_size_test, shuffle=False)

test(test_loader)