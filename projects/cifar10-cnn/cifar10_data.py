"""CIFAR-10 loaders with a reproducible training/validation split."""

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


def get_data_loaders(root="./data", batch_size=64):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    full_train = datasets.CIFAR10(
        root=root, train=True, download=True, transform=transform
    )
    test = datasets.CIFAR10(
        root=root, train=False, download=True, transform=transform
    )
    train, val = random_split(
        full_train, [45000, 5000], generator=torch.Generator().manual_seed(42)
    )
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, val_loader, test_loader
