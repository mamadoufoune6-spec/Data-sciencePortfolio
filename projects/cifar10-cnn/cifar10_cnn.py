"""Original CIFAR-10 model; use ../train_classifier.py for training."""

import torch
import torch.nn as nn
import torch.optim as optim
from cifar10_data import get_data_loaders


class CIFAR10CNN(nn.Module):
    def __init__(self):
        super(CIFAR10CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(2048, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 2048)
        x = self.fc1(x)
        return x


lr = 0.1
model = CIFAR10CNN()
opt = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
criterion = nn.CrossEntropyLoss()
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)


def val_loader(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    val_loss /= len(val_loader)
    accuracy = 100 * correct / total
    return val_loss, accuracy


if __name__ == "__main__":
    train_loader, val_loader, test_loader = get_data_loaders()
    images, labels = next(iter(train_loader))
    images = images.to(device)
    with torch.no_grad():
        logits = model(images)
    print("Dataset sizes:", len(train_loader.dataset), len(val_loader.dataset), len(test_loader.dataset))
    print("Images:", images.shape, images.dtype)
    print("Labels:", labels.shape, labels.dtype)
    print("Logits:", logits.shape)
    assert logits.shape == (images.size(0), 10)
