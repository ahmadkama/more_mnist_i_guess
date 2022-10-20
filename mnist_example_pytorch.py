import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision

transform = torchvision.transforms.Compose([
                           torchvision.transforms.ToTensor(),
                           torchvision.transforms.Normalize(0.1307, 0.3081)])

train_data = torchvision.datasets.MNIST('./datafiles/', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=100, shuffle=True)

test_data = torchvision.datasets.MNIST('./datafiles/', train=False, download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=1000, shuffle=True)


class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.hidden = nn.Linear(784, 100)
        self.output = nn.Linear(100, 10)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.hidden(x)
        x = F.relu(x)
        return self.output(x)


def train(model, train_loader, optimizer, epoch):
    model.train()
    total_loss = 0
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = nn.CrossEntropyLoss()(outputs, targets)
        total_loss += loss
        loss.backward()
        optimizer.step()

        if batch_idx % 100 == 0:
            print('Epoch: {} {}/{} Training loss: {:.6f}'.format(
                epoch,
                batch_idx * len(inputs),
                len(train_loader.dataset),
                loss))

    print('Training loss: {:.6f}'.format(total_loss / len(train_loader.dataset) * len(inputs)))


def test(model, test_loader):
    model.eval()
    loss = 0
    correct = 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            outputs = model(inputs)
            loss += nn.CrossEntropyLoss()(outputs, targets)
            predictions = outputs.argmax(dim=1, keepdim=True)
            correct += predictions.eq(targets.view_as(predictions)).sum()

    loss = loss / len(test_loader.dataset) * len(inputs)

    print('Test loss: {:.6f}; Test accuracy: {}/{} ({:.1f}%)\n'.format(
        loss,
        correct,
        len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


if __name__ == '__main__':
    model = Network()
    optimizer = optim.Adam(model.parameters())

    for epoch in range(1, 5):
        train(model, train_loader, optimizer, epoch)
        test(model, test_loader)
