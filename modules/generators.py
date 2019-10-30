import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from torch.distributions import Categorical
import numpy as np
import types
from copy import copy
from utils import DDICT


class ConditionalBatchNorm1d(nn.Module):
    r"""Conditional BatchNorm

        Args:
            num_features (int): number of features in input
            num_classes (int): number of classe among for labels in input

        Attributes:
            embed (Tensor): embedding of labels to scaling matrix `gamma` and bias matrix `beta`

        Shape:
            - Inputs:
                inputs: 2-tuple such that inputs = (x, labels), with
                    x: FloatTensor of shape `(batch_size, num_features)`
                    labels: LongTensor of size `(batch_size)`
            - Outputs: FloatTensor of shape `(batch_size, num_features)`, i.e. same as x
    """
    def __init__(self, num_features, num_classes):
        super().__init__()
        self.num_features = num_features
        self.num_classes = num_classes

        self.bn = nn.BatchNorm1d(num_features, affine=False)
        self.embed = nn.Embedding(num_classes, num_features * 2)

        # Scale and biases
        self.embed.weight.data[:, :num_features].normal_(1, 1.0 / np.sqrt(num_features))
        self.embed.weight.data[:, num_features:].zero_()

    def forward(self, inputs):
        x, labels = inputs
        outputs = self.bn(x)
        gamma, beta = self.embed(labels).chunk(2, 1)
        outputs = gamma.view(-1, self.num_features) * outputs + beta.view(-1, self.num_features)
        return outputs

    def __repr__(self):
        fmt_str = self.__class__.__name__
        fmt_str += '(num_features={num_features}, num_classes={num_classes})'.format(**self.__dict__)
        return fmt_str


class Generator(nn.Module):
    r"""Generic generator that will be inherited by all generators.

        Args:
            num_features (int): number of inputs features
            n_layers (int): number of hidden layers (each one with ConditionalBatchNorm1d)
            n_hiddens (int): number of hidden neurons
            p_dropout (float): dropout rate

        Shape:
            - Inputs:
                inputs: 2-tuple such that inputs = (x, labels), with
                    x: FloatTensor of size `(batch_size, in_feaures)`
                    labels: LongTensor of size `(batchsize, 1)` indicating which feature has to be predicted
            - Output: Not implemented
    """
    def __init__(self, num_features, n_layers, n_hiddens, p_dropout=0.0):
        super().__init__()
        self.num_features = num_features
        self.n_layers = n_layers
        self.n_hiddens = n_hiddens
        self.p_dropout = p_dropout

        # Linear layers and ConditionalBatchNorm
        self.w, self.cb, self.dr = nn.ModuleList(), nn.ModuleList(), nn.ModuleList()
        for k in range(n_layers):
            self.w.append(nn.Linear(num_features if k == 0 else n_hiddens, n_hiddens))
            self.cb.append(ConditionalBatchNorm1d(n_hiddens, num_features))
            if p_dropout > 0.0:
                self.dr.append(nn.Dropout(p=self.p_dropout))

        self.w_out = None
        self.criterion = None

    def forward(self, inputs):
        x, labels = inputs
        x = x.view(-1, self.num_features)

        # mask with a zero in correspondance to label in `labels`
        mask = torch.ones_like(x)
        mask.scatter_(-1, labels.view(-1, 1), 0.0)

        x = x * mask
        for l, (w, cb) in enumerate(zip(self.w, self.cb)):
            x = F.relu(cb((w(x), labels)))
            if self.p_dropout > 0.0:
                x = self.dr[l](x)
        return self._sample_outputs(x)

    def get_targets(self, x, labels):
        """Returns the targets, i.e. the features corresponding to the labels
        """
        idx = torch.arange(0, len(labels))
        return x[idx, labels].view(-1, 1)

    def get_dataloader(self, dl, idx_feature):
        """Returns a dataloader that is the same as `dataloader` except that the feature `i` is sampled from the generator

            Args:
                dl (DataLoader): base data loader
                idx_feature (int): index of the feature that is replaced
        """
        new_loader = DataLoader(dataset=GenDataset(self, dl.dataset, idx_feature), batch_size=dl.batch_size)
        return new_loader

    def _sample_outputs(self, x):
        raise NotImplementedError

    def sample_features(self, inputs):
        """Samples with no noise. Used by `test_generator`.
        """
        raise NotImplementedError

    def training_loss(self, x, labels):
        raise NotImplementedError


class GeneratorAvg(Generator):
    r"""Generator that as prediction of a feature outputs the average of all other features

        Args:
            num_features (int): number of inputs features

        Shape:
            - Inputs:
                inputs: 2-tuple such that inputs = (x, labels), with
                    x: FloatTensor of size `(batch_size, in_feaures)`
                    labels: LongTensor of size `(batchsize, 1)` indicating which feature has to be predicted
            - Output: FloatTensor of size `(batch_size, 1)`
    """
    def __init__(self, num_features):
        super().__init__(num_features, 0, 0, 0)
        self.id = nn.Parameter(torch.randn(1))  # needed for backward compatibility of Generator interface (so that gen.parameters() is not empty)

    def _sample_outputs(self, x):
        """The features corresponding to `labels` have been masked out.
            Predicts the average of the other features.
        """
        mu = x.sum(-1) / (self.num_features - 1)
        sd = x.std(-1) / np.sqrt(self.num_features - 1)
        eps = torch.randn(mu.size()) / 20
        return mu + sd * eps

    def sample_features(self, inputs):
        """This is redundant wrt forward. It's just here to have a coherent interface with GeneratorClassify
        """
        self.eval()
        with torch.no_grad():
            outputs = self(inputs)
        return outputs


class GeneratorOracle(Generator):
    r"""Generator that from a sample of a set of correlated features with known correlation `rho`, predicts an output with same correlation

        Args:
            num_features (int): number of inputs features
            gaussian (bool): whether the variables are assumed to be gaussian or uniform
            rho (float): pairwise correlation between variables (features)
            normalize (True): whether the features are normalized

        Shape:
            - Inputs:
                inputs: 2-tuple such that inputs = (x, labels), with
                    x: FloatTensor of size `(batch_size, in_feaures)`
                    labels: LongTensor of size `(batchsize, 1)` indicating which feature has to be predicted
            - Output: FloatTensor of size `(batch_size, 1)`
    """
    def __init__(self, num_features, gaussian=False, rho=0.5, normalize=True):
        super().__init__(num_features, 0, 0, 0)
        self.id = nn.Parameter(torch.randn(1))  # needed for backward compatibility of Generator interface (so that gen.parameters() is not empty)

        self.gaussian = gaussian
        self.rho = rho
        self.normalize = normalize

        # Normalizations
        if gaussian:
            self.X_mu, self.X_sd = 0.0, 1.0
        else:
            self.X_mu = 0.5 * np.sqrt(self.rho) + 0.5 * np.sqrt(1 - self.rho)
            self.X_sd = 1 / np.sqrt(12)

    def _sample_outputs(self, inputs):
        """The features corresponding to `labels` have been masked out.
            Predicts the average of the other features.
        """
        N, P = inputs.size()

        # Estimate of correlation factor
        corrcoef = inputs.sum(-1) / (P - 1)
        if self.normalize:
            corrcoef = self.X_sd * corrcoef + self.X_mu

        # Generate feature
        if self.gaussian:
            X = corrcoef * np.sqrt(self.rho) + np.sqrt(1 - self.rho) * torch.randn(N)
        else:
            X = corrcoef * np.sqrt(self.rho) + np.sqrt(1 - self.rho) * torch.rand(N)

        if self.normalize:
            X = (X - self.X_mu) / self.X_sd

        return X

    def sample_features(self, inputs):
        """This is redundant wrt forward. It's just here to have a coherent interface with GeneratorClassify
        """
        self.eval()
        with torch.no_grad():
            outputs = self(inputs)
        return outputs


class GeneratorRegress(Generator):
    r"""Generator that outputs regression prediction
        Architecture is a 2-layer network, training loss is Huber loss

        Args:
            num_features (int): number of inputs features
            n_layers (int): number of hidden layers (each one with ConditionalBatchNorm1d)
            n_hiddens (int): number of hidden neurons
            p_dropout (float): dropout rate

        Shape:
            - Inputs:
                inputs: 2-tuple such that inputs = (x, labels), with
                    x: FloatTensor of size `(batch_size, in_feaures)`
                    labels: LongTensor of size `(batchsize, 1)` indicating which feature has to be predicted
            - Output: FloatTensor of size `(batch_size, 1)`
    """
    def __init__(self, num_features, n_layers, n_hiddens, p_dropout=0.0):
        super().__init__(num_features, n_layers, n_hiddens, p_dropout)

        self.w_out = nn.Linear(n_hiddens, 2)  # mu, log_var
        self.criterion = nn.SmoothL1Loss()  # Huber loss

    def _sample_outputs(self, x):
        """Samples from a Gaussian distribution with mu=inputs[:,0] and log_var=inputs[:,1]
            using the reparametrization trick
        """
        inputs = self.w_out(x)

        mu, log_var = inputs.chunk(2, dim=-1)
        eps = torch.randn(mu.size())
        return mu + torch.exp(log_var / 2) * eps

    def sample_features(self, inputs):
        """This is redundant wrt forward. It's just here to have a coherent interface with GeneratorClassify
        """
        self.eval()
        with torch.no_grad():
            outputs = self(inputs)
        return outputs

    def training_loss(self, x, labels):
        """Huber loss
        """
        outputs = self((x, labels))
        targets = self.get_targets(x, labels)
        return self.criterion(outputs, targets)


class GeneratorClassify(Generator):
    r"""Generator that outputs classification prediction (softmax)
        Architecture is a 2-layer network, training loss is cross-entropy

        Args:
            num_features (int): number of inputs features
            n_layers (int): number of hidden layers (each one with ConditionalBatchNorm1d)
            n_hiddens (int): number of hidden neurons
            num_bins (int): number of bins, i.e. number of softmax outputs
            init_dataset (torch.FloatTensor, Dataset or DataLoader): a set of input samples representative of the dataset,
                necessary to compute the bins for quantization
            beta (float): pseudo-temperature for sampling over bins
            p_dropout (float): dropout rate

        Shape:
            - Inputs:
                inputs: 2-tuple such that inputs = (x, labels), with
                    x: FloatTensor of size `(batch_size, in_feaures)`
                    labels: LongTensor of size `(batchsize, 1)` indicating which feature has to be predicted
            - Output: FloatTensor of size `(batch_size, num_bins)`
    """
    def __init__(self, num_features, n_layers, n_hiddens, num_bins, init_dataset, beta=1.0, p_dropout=0.0):
        super().__init__(num_features, n_layers, n_hiddens, p_dropout)

        self.num_bins = num_bins
        self.beta = beta
        self.w_out = nn.Linear(n_hiddens, num_bins)
        self.criterion = nn.CrossEntropyLoss()

        # Initialize binning
        if isinstance(init_dataset, DataLoader):
            self.init_dataset = init_dataset.dataset[:][0]
        elif isinstance(init_dataset, Dataset):
            self.init_dataset = init_dataset[:][0]
        else:
            self.init_dataset = init_dataset

        self.bin_edges, self.bin_centers, self.bin_widths = self._quantization_binning(self.init_dataset, num_bins)
        self.bin_centers = torch.FloatTensor(self.bin_centers)
        self.bin_widths = torch.FloatTensor(self.bin_widths)

    def _quantization_binning(self, data, num_bins):
        """Quantize the inputs and computes binning, assuming that all input features have same distribution

            Shape:
            - Outputs:
                bin_edges: array of size `(num_bins + 1, num_features)`, edges of bins for each feature
                bin_centers: array of size `(num_bins, num_features)`, ceters of bins for each feature
        """
        qtls = np.arange(0.0, 1.0 + 1 / num_bins, 1 / num_bins)
        bin_edges = np.quantile(data, qtls, axis=0)  # (num_bins + 1, num_features)
        bin_widths = np.diff(bin_edges, axis=0)
        bin_centers = bin_edges[:-1, :] + bin_widths / 2  # ()
        return bin_edges, bin_centers, bin_widths

    def _quantize(self, inputs, labels):
        quant_inputs = np.zeros(inputs.shape[0])
        for i, (x, l) in enumerate(zip(inputs.cpu(), labels)):
            quant_inputs[i] = np.digitize(x, self.bin_edges[:, l])
        quant_inputs = quant_inputs.clip(1, self.num_bins) - 1  # Clip edges
        return torch.LongTensor(quant_inputs).to(inputs.device)

    def sample_features(self, inputs):
        """Samples with no noise
        """
        self.eval()
        with torch.no_grad():
            x, labels = inputs
            logits = self(inputs)
            sampled_bins = Categorical(logits=self.beta * logits).sample()
            samples = self.bin_centers[sampled_bins, labels] + (torch.rand(len(sampled_bins)) - 0.5) * self.bin_widths[sampled_bins, labels]
        return samples.to(x.device).view(-1, 1)

    def _sample_outputs(self, x):
        return self.w_out(x)

    def training_loss(self, x, labels):
        """Computes cross-entropy loss from classification output
        """
        outputs = self((x, labels))
        targets = self.get_targets(x, labels)

        quant_targets = self._quantize(targets, labels).view(-1)
        return self.criterion(outputs, quant_targets)


def train_generator(generator, dataloader, optimizer, features_list=None, log_times=5):
    '''Trains a generator model

        Args:
            generator (Generator): a generator object
            dataloader (Dataloader): a dataloader object
            optimizer (optim.optimizer): optimizer used to train `generator`
            feature_list (list): list of features to which training has to be restricted
            log_times (int): how many times the training will be logged
    '''
    generator.train()
    device = next(generator.parameters()).device

    # In case we're using nn.DataParallel
    if isinstance(generator, nn.DataParallel):
        generator_loss = generator.module.training_loss
    else:
        generator_loss = generator.training_loss

    # Subset of features to train on
    if features_list is None:
        # Train to output all features
        features_list = list(range(generator.num_features))
    features_list = torch.LongTensor(features_list)

    mean_loss = 0.0
    for batch_idx, data in enumerate(dataloader):
        if isinstance(data, tuple) or isinstance(data, list):
            data = data[0]
        data = data.to(device)

        # Generate labels at random
        rIdx = torch.randint(0, len(features_list), (data.shape[0],)).to(device)
        labels = features_list.index_select(0, rIdx)

        optimizer.zero_grad()
        loss = generator_loss(data, labels)
        loss.backward()
        optimizer.step()

        mean_loss += loss.item() / len(data)
        if log_times > 0 and batch_idx % (len(dataloader) // log_times) == 0:
            print('   training progress: {}/{} ({:.0f}%)\tloss: {:.6f}'.format(
                batch_idx * len(data), len(dataloader.dataset), 100. * batch_idx / len(dataloader), loss.item()))

    return mean_loss


def test_generator(generator, criterion, dataloader, test_all_features=False):
    '''Test generator model

        Args:
            test_all_features (bool): if True, test all features for all samples as outputs,
                otherwise only test one feature at random per sample
    '''
    generator.eval()
    device = next(generator.parameters()).device

    # In case we're using nn.DataParallel
    if isinstance(generator, nn.DataParallel):
        generator_get_targets = generator.module.get_targets
        generator_sample_features = generator.module.sample_features
    else:
        generator_get_targets = generator.get_targets
        generator_sample_features = generator.sample_features

    test_loss = 0.0
    with torch.no_grad():
        for data in dataloader:
            data = data[0].to(device)

            if test_all_features:
                # Generate labels by running all features for all samples
                labels = torch.arange(0, data.shape[-1]).repeat(data.shape[0], 1).view(-1).to(device)
                idx = torch.arange(0, data.shape[0]).repeat(data.shape[-1], 1).t().reshape(-1).to(device)
            else:
                # Generate labels )t random
                labels = torch.randint(0, data.shape[-1], (data.shape[0],)).to(device)
                idx = torch.arange(0, len(labels)).to(device)

            targets = generator_get_targets(data[idx], labels)
            outputs = generator_sample_features((data[idx], labels))
            test_loss += criterion(outputs, targets).item()

    test_loss /= len(dataloader)  # loss function already averages over batch size
    return test_loss


class GenDataset(Dataset):
    r"""Generator dataset: replaces one feature of the input dataset with one generated by a generator
        Args:
            generator: generator trained to generate inputs features
            dataset (Dataset): dataset whose features are going to be replaced
            idx_feature (int): feature that will be replaced by the generator

        Attributes:
            idx_feature: feature which is being replaced by the generator

        Notes:
            - the replacement feature is sampled only once (at initialization)
            - if you want to resample replacement features, call `resample_replaced_feature()`

    """
    def __init__(self, generator, dataset, idx_feature):
        self.generator = generator
        self.dataset = dataset
        self.idx_feature = idx_feature
        self.resample_replaced_feature()

    def resample_replaced_feature(self):
        device = next(self.generator.parameters()).device
        replaced_feature = []
        for data in DataLoader(self.dataset, batch_size=256, shuffle=False):
            data = data[0].to(device)
            labels = data.new_full((data.shape[0],), self.idx_feature, dtype=torch.long)
            replaced_feature.append(self.generator.sample_features((data, labels)).view(-1))

        self._replaced_features = torch.cat(replaced_feature).to(data.device)

    def __getitem__(self, index):
        data = self.dataset.__getitem__(index)

        if len(data) > 1:
            data, target = data
        else:
            data, target = data[0], None

        r_data = data.new_empty(data.size())
        r_data.copy_(data)
        r_data[..., self.idx_feature] = self._replaced_features[index]
        return r_data, target

    def __len__(self):
        return len(self.dataset)


def generator_from_data(dataset, generator_type='regress', features_list=None, n_epochs=100, n_layers=3, n_hiddens=200, p_dropout=0, num_bins=100, training_args=None):
    """NOTE: Training epochs `n_epochs` should scale with the number of features.
    """
    if generator_type == 'oracle':
        n_features = dataset[0][0].shape[-1]
        generator = GeneratorOracle(n_features, gaussian=dataset.gaussian, rho=dataset.rho, normalize=dataset.normalize)

        return generator, None

    else:  # Generator needs to be trained

        # All default training arguments are hidden here
        default_args = DDICT(
            optimizer='Adam',
            batch_size=128,
            lr=0.003,
            lr_step_size=20,
            lr_decay=0.5,
            num_bins=10,
        )

        # Custom training arguments
        args = default_args
        if training_args is not None:
            for k in training_args:
                args[k] = training_args[k]

        # Data
        n_features = dataset[0][0].shape[-1]
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

        if generator_type == 'classify':
            generator = GeneratorClassify(n_features, n_layers, n_hiddens, num_bins=num_bins, init_dataset=dataset)
        elif generator_type == 'regress':
            generator = GeneratorRegress(n_features, n_layers, n_hiddens)
        else:
            raise ValueError('generator_type has to be classify or regress')

        optimizer = getattr(optim, args.optimizer)(generator.parameters(), lr=args.lr)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_decay)

        tr_loss = []
        for epoch in range(n_epochs):
            tr_loss += [train_generator(generator, dataloader, optimizer, features_list, log_times=0)]
            scheduler.step()

        return generator, tr_loss
