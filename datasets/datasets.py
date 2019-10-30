import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np


def load_mnist(batch_size=200, conv_net=False, num_workers=1):
    '''Load the MNIST dataset

        Args:
            conv_net: set to `True` if the dataset is being used with a conv net (i.e. the inputs have to be 3d tensors and not flattened)
    '''
    DIR_DATASET = '~/data/mnist'

    transform_list = [
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))]

    if not conv_net:
        transform_list.append(transforms.Lambda(lambda x: x.view(x.size(1) * x.size(2))))

    transform = transforms.Compose(transform_list)

    trainset = datasets.MNIST(DIR_DATASET, train=True, download=True, transform=transform)
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    testset = datasets.MNIST(DIR_DATASET, train=False, transform=transform)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    input_shape = trainset.train_data[0].shape

    if conv_net:
        # Channels, width, height
        input_shape = tuple(input_shape[-1:] + input_shape[:-1])
    else:
        input_shape = np.prod(input_shape)

    return train_loader, test_loader, input_shape


def load_fashion(batch_size=200, conv_net=False, num_workers=1):
    '''Load the fashion MNIST dataset

        Args:
            conv_net: set to `True` if the dataset is being used with a conv net (i.e. the inputs have to be 3d tensors and not flattened)
    '''
    DIR_DATASET = '~/data/fashion'

    transform_list = [
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))]

    if not conv_net:
        transform_list.append(transforms.Lambda(lambda x: x.view(x.size(1) * x.size(2))))

    transform = transforms.Compose(transform_list)

    trainset = datasets.FashionMNIST(DIR_DATASET, train=True, download=True, transform=transform)
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    testset = datasets.FashionMNIST(DIR_DATASET, train=False, transform=transform)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    input_shape = trainset.train_data[0].shape

    if conv_net:
        # Channels, width, height
        input_shape = tuple(input_shape[-1:] + input_shape[:-1])
    else:
        input_shape = np.prod(input_shape)

    return train_loader, test_loader, input_shape


def load_sklearndata(namedataset='iris', batch_size=100, test_size=0.1, seed=123, z_score=True, xname=None, yname=None, **kwargs):
    r"""Loads any of the sklearn standard datasets, which includes several UCI datasets

        Args:
            datase_name (string): the name of the dataset to load
            xname (strin): name of inputs x (if it's None, guess `data`)
            yname (strin): name of targets y (if it's None, guess `target`)
            kwargs (dict): arguments to pass to sklearn dataset loader
    """
    from sklearn import datasets as skdatasets

    if hasattr(skdatasets, namedataset):
        name = namedataset
    elif hasattr(skdatasets, 'load_' + namedataset):
        name = 'load_' + namedataset
    elif hasattr(skdatasets, 'make_' + namedataset):
        name = 'make_' + namedataset
    else:
        raise ValueError('Dataset {} not recognized'.format(namedataset))

    print('Loading scikit learn dataset {}'.format(name))
    dataset = getattr(skdatasets, name)(**kwargs)

    if xname is None:
        xname = 'data'
    if yname is None:
        yname = 'target'

    data = dataset[xname]
    targets = dataset[yname]

    # Split in trainind and test
    if isinstance(test_size, float):
        if test_size > 1.0 or test_size < 0.0:
            raise ValueError('test_size must be integer or a float between 0.0 and 1.0')
        else:
            test_size = int(len(data) * test_size)
    elif isinstance(test_size, int):
        if test_size >= len(data) or test_size < 0:
            raise ValueError('integer test_size must be between 0 and {}'.format(len(data)))

    # Random number generator
    rng = np.random.RandomState(seed)

    # Permutation indices:
    perm = rng.permutation(len(data))
    ind_train = perm[test_size:]
    ind_test = perm[:test_size]

    train_data, train_targets = data[ind_train], targets[ind_train]
    test_data, test_targets = data[ind_test], targets[ind_test]

    # z-score according to training split
    if z_score:
        mu = np.mean(data[ind_train], 0)
        sd = np.std(data[ind_train], 0) + 1e-6
        train_data = (train_data - mu) / sd
        test_data = (test_data - mu) / sd

    # Convert to torch.Tensor
    train_data = torch.FloatTensor(train_data)
    test_data = torch.FloatTensor(test_data)

    if np.issubdtype(train_targets.reshape(-1, 1)[0][0], np.integer):
        train_targets = torch.LongTensor(train_targets)
        test_targets = torch.LongTensor(test_targets)
    else:
        train_targets = torch.FloatTensor(train_targets)
        test_targets = torch.FloatTensor(test_targets)

    trainset = torch.utils.data.TensorDataset(train_data, train_targets)
    testset = torch.utils.data.TensorDataset(test_data, test_targets)

    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=True)

    input_shape = data.shape[-1]

    return train_loader, test_loader, input_shape
