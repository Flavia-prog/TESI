import numpy as np
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


def get_mnist(data_root: str = "./data"):
    transform = transforms.ToTensor()
    train = datasets.MNIST(root=data_root, train=True, download=True, transform=transform)
    test = datasets.MNIST(root=data_root, train=False, download=True, transform=transform)
    return train, test


def get_train_subset(train_dataset, max_samples: int = 2000, seed: int = 0):
    max_samples = min(max_samples, len(train_dataset))
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(train_dataset), size=max_samples, replace=False)
    return Subset(train_dataset, idx.tolist())


def split_iid(dataset, num_clients: int = 2, seed: int = 0):
    rng = np.random.default_rng(seed)
    indices = np.arange(len(dataset))
    rng.shuffle(indices)
    splits = np.array_split(indices, num_clients)
    return [Subset(dataset, split.tolist()) for split in splits]


def make_loaders(client_datasets, test_dataset, batch_size: int = 64):
    client_loaders = [DataLoader(ds, batch_size=batch_size, shuffle=True) for ds in client_datasets]
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)
    return client_loaders, test_loader
