from collections import Counter

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets


# Load and Preprocess data.
# Loading: If the dataset is not in the given directory, it will be downloaded.
# Preprocessing: This includes normalizing each channel and data augmentation by random cropping and horizontal flipping
def load_data(split, dataset_name, datadir, corrupt_prob=1.0):

    if dataset_name == 'MNIST':
        normalize = transforms.Normalize(mean=[0.131], std=[0.289])
    else:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    tr_transform = transforms.Compose([transforms.Resize(32), transforms.ToTensor(), normalize])
    val_transform = transforms.Compose([transforms.Resize(32), transforms.ToTensor(), normalize])

    if dataset_name == 'CIFAR10':
        # if CIFAR dataset
        if split == 'train':
            dataset = datasets.CIFAR10(root=datadir, train=True, download=True,
                                       transform=tr_transform)
        else:
            dataset = datasets.CIFAR10(root=datadir, train=False, download=True,
                                       transform=val_transform)
    elif dataset_name == 'CIFAR10RandomLabels':
        if split == 'train':
            dataset = CIFAR10RandomLabels(root=datadir, train=True, download=True,
                                          transform=tr_transform, corrupt_prob=corrupt_prob)
        else:
            dataset = CIFAR10RandomLabels(root=datadir, train=False, download=True,
                                          transform=tr_transform, corrupt_prob=corrupt_prob)
    elif dataset_name == 'MNIST':
        if split == 'train':
            dataset = datasets.MNIST(root=datadir, train=True, transform=tr_transform,
                                     download=True)
        else:
            dataset = datasets.MNIST(root=datadir, train=False, transform=val_transform,
                                     download=True)
    else:
        raise ValueError('not a valid dataset choice.')
    return dataset


class CIFAR10RandomLabels(datasets.CIFAR10):
    """CIFAR10 dataset, with support for randomly corrupt labels.
    Params
    ------
    corrupt_prob: float
      Default 1.0. The probability of a label being replaced with
      random label.
    num_classes: int
      Default 10. The number of classes in the dataset.
    """

    def __init__(self, corrupt_prob=1.0, num_classes=10, **kwargs):
        super(CIFAR10RandomLabels, self).__init__(**kwargs)
        self.n_classes = num_classes
        if corrupt_prob > 0:
            self.corrupt_labels(corrupt_prob)

    def corrupt_labels(self, corrupt_prob):
        labels = np.array(self.targets)
        np.random.seed(12345)
        mask = np.random.rand(len(labels)) <= corrupt_prob
        rnd_labels = np.random.choice(self.n_classes, mask.sum())
        labels[mask] = rnd_labels
        # we need to explicitly cast the labels from npy.int64 to
        # builtin int type, otherwise pytorch will fail...
        labels = [int(x) for x in labels]

        self.targets = labels


def CIFARSubset(args, batchsize=64, **kwargs):
    # loading data
    if args.randomlabels == True:
        train_dataset = load_data('train', 'CIFAR10RandomLabels', args.datadir)
    else:
        train_dataset = load_data('train', 'CIFAR10', args.datadir)

    val_dataset = load_data('val', 'CIFAR10', args.datadir)
    train_subset = torch.utils.data.Subset(train_dataset, list(range(0, args.trainingsetsize)))

    train_loader = DataLoader(train_subset, batch_size=batchsize, shuffle=True, **kwargs)
    val_loader = DataLoader(val_dataset, batch_size=batchsize, shuffle=True, **kwargs)

    return train_loader, val_loader


def MNIST(args, batchsize=64, **kwargs):
    # loading data
    if args.randomlabels == True:
        train_dataset = load_data('train', 'MNIST', args.datadir)
    else:
        train_dataset = load_data('train', 'MNIST', args.datadir)

    val_dataset = load_data('val', 'MNIST', args.datadir)

    train_loader = DataLoader(train_dataset, batch_size=batchsize, shuffle=True, **kwargs)
    val_loader = DataLoader(val_dataset, batch_size=batchsize, shuffle=True, **kwargs)

    return train_loader, val_loader


def get_classbalance(dataset, setsize, kwargs):
    train2_loader = DataLoader(dataset, batch_size=setsize, shuffle=True, **kwargs)
    for data, target in train2_loader:
        return Counter(target.numpy())
