import math
import torch
import numpy as np
import torch.utils.data


class ToyDataset(torch.utils.data.Dataset):
    """Implementation of a toy dataset as described in https://arxiv.org/pdf/1606.07892.pdf sec 5.1 5.2.
    """

    def __init__(self, opt, shuffle_targets = False, data = None, targets = None, seed = 31):
        # Random number generator
        self.rng = np.random.RandomState(seed)

        self.shuffle_targets = shuffle_targets
        self.data            = data
        self.targets         = targets
        self.features        = [ 'x%d'%i for i in range(opt.Xdim)]
        self.fea_groundtruth = [0, 1]

        if self.data is None:
            self.data = torch.randn(opt.numSamples, opt.Xdim)

        if self.targets is None:
            Z = torch.randn(opt.numSamples)
            if opt.Yfunction == 'linear':
                self.targets = self.data[:,0] + Z
            elif opt.Yfunction == 'sine':
                self.targets = 5.0 * torch.sin(4*math.pi*(self.data[:,0]**2 + self.data[:,1]**2)) + 0.25 * Z
            self.targets = self.targets.view(opt.numSamples,1)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.shuffle_targets:
            y_index = self.rng.randint(len(self.targets))
        else:
            y_index = index
        return self.data[index], self.targets[y_index]

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())

        fmt_str += '=== X === \n'
        t = self.data.data if isinstance(self.data, torch.Tensor) else self.data
        s = '{:8s} [{:.4f} , {:.4f}] m+-s = {:.4f} +- {:.4f}'
        si = 'x'.join(map(str, t.shape if isinstance(t, np.ndarray) else t.size()))
        fmt_str += s.format(si, t.min(), t.max(), t.mean(), t.std()) + '\n'

        fmt_str += '=== Y === \n'
        t = self.targets.data if isinstance(self.targets, torch.Tensor) else self.targets
        s = '{:8s} [{:.4f} , {:.4f}] m+-s = {:.4f} +- {:.4f}'
        si = 'x'.join(map(str, t.shape if isinstance(t, np.ndarray) else t.size()))
        fmt_str += s.format(si, t.min(), t.max(), t.mean(), t.std()) + '\n'

        return fmt_str

    def __len__(self):
        return len(self.data)

    def get_feature_names(self):
        return self.features

    def get_groundtruth_features(self):
        return self.fea_groundtruth


class LiangDataset(torch.utils.data.Dataset):
    """Generates data from the simulation study in Liang et al, JASA 2017
        Sourced from : https://github.com/tansey/hrt/blob/master/benchmarks/liang/sim_liang.py
        N = 500 # total number of samples
        P = 500 # number of features
        S = 40 # number of signal features
        T = 100 # test sample size
    """
    def __init__(self, numSamples, Xdim, seed, S=40, betas=None, data=None, targets=None, shuffle_targets=False, targets_mu=1.0, targets_sd=1.0):
        # Random number generator
        self.rng = np.random.RandomState(seed)

        self.shuffle_targets = shuffle_targets
        self.data            = data
        self.betas           = betas
        self.targets         = targets
        self.features        = [ 'x%d'%i for i in range(Xdim)]
        self.fea_groundtruth = [i for i in range(S)]

        if betas is None:
            self.betas = self.compute_betas(S)

        if self.data is None and self.targets is None:
            self.data    = self.make_X(numSamples, Xdim)
            self.targets = self.make_targets(self.data, S) # may modify X in-place
            self.data = torch.from_numpy(self.data).type('torch.FloatTensor')
            self.targets = torch.from_numpy(self.targets).type('torch.FloatTensor')
            self.targets = self.targets.view(numSamples,1)

        self.targets_mu, self.targets_sd = targets_mu, targets_sd

    def compute_betas(self, S):
        w0 = self.rng.normal(1, size=S//4)
        w1 = self.rng.normal(2, size=S//4)
        w2 = self.rng.normal(2, size=S//4)
        w21 = self.rng.normal(1, size=(1,S//4))
        w22 = self.rng.normal(2, size=(1,S//4))
        return [w0, w1, w2, w21, w22]

    def make_X(self, N, P):
        X = (self.rng.normal(size=(N,1)) + self.rng.normal(size=(N,P))) / 2.
        return X

    def make_targets(self, X, S):
        N, P = X.shape
        w0, w1, w2, w21, w22 = self.betas
        y = X[:,0:S:4].dot(w0) + X[:,1:S:4].dot(w1) + np.tanh(w21*X[:,2:S:4] + w22*X[:,3:S:4]).dot(w2) + self.rng.normal(0, 0.5, size=N)
        return y

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.shuffle_targets:
            y_index = self.rng.randint(len(self.targets))
        else:
            y_index = index
        return self.data[index], (self.targets[y_index] - self.targets_mu) / self.targets_sd

    def __len__(self):
        return len(self.data)
    def get_feature_names(self):
        return self.features

    def get_groundtruth_features(self):
        return self.fea_groundtruth

class LiangSwitchingDataset(LiangDataset):
    def make_targets(self, X, S):
        # NOTE modifies X in-place
        N, P = X.shape
        R = S//4 # number of regions to switch between
        w0, w1, w2, w21, w22 = self.betas
        w21, w22 = w21.squeeze(), w22.squeeze()
        r = self.rng.choice(R, replace=True, size=N)
        y = np.zeros(N)
        Z = np.zeros((N,R)) # one hot indication of region
        truth = np.zeros((R,P-R), dtype=int)
        for i in range(R):
            y[r == i] = (X[r == i,i*4] * w0[i] +
                         X[r == i,i*4+1] * w1[i] +
                         w2[i] * np.tanh(w21[i]*X[r==i,i*4+2] + w22[i]*X[r==i,i*4+3]))
            Z[r==i, i] = 1
            truth[i] = np.concatenate([np.zeros(i*4), np.ones(4), np.zeros((R-i-1)*4), np.zeros(P-5*R)])
        y += self.rng.normal(0, 0.5, size=N)
        assert P >= S+R, 'Need high enough X dimension to have S used features, and R one-hots'
        # overwrite (unused) last R features in X with one hot region indicators.
        X[:, -R:] = Z
        # Return just y
        return y


class SinExpDataset(torch.utils.data.Dataset):
    """Complex multivariate model from https://www.padl.ws/papers/Paper%2012.pdf section 5.3
        y = sin(x_1 * (x_1 + x_2)) * cos(x_3 + x_4 * x_5) * sin(exp(x_5) + exp(x_6) - x_2) + eps
        with P = 50, in the original paper, sampled from uniform distribution,
        and eps sampled from a zero-centered Gaussian with variance such that the SNR = 2

        Args:
            opt.Xdim (int): number of features
            opt.numSamples (int): number of samples
            shuffle_targets (bool): whether y should be reshuffled to be decorrelated from X
            snr (float): signal to noise ratio of y
            gaussian (bool): sample covariates from gaussian or uniform
            rho (float): correlation coefficient between pairs of covariates:
                x_i = sqrt(rho) * z + sqrt(1 - rho) * randn(0, 1), with z common to all x_i
            sigma (float): noise amplitude
            seed (int): random seed
    """
    def __init__(self, n_samples=125, n_features=50, shuffle_targets=False, data=None, targets=None, betas=None,
                 rho=0.5, gaussian=False, normalize=True, sigma=None, seed=31):

        self.n_samples = n_samples
        self.n_features = n_features
        self.shuffle_targets = shuffle_targets
        self.rho = rho
        self.sigma = sigma
        self.gaussian = gaussian
        self.normalize = normalize
        self.seed = seed

        # Interface stuff
        self.fea_groundtruth = list(range(6))
        self.features = ['x{}'.format(i) for i in range(self.n_features)]
        self.betas = None

        # Random number generator
        self.rng = np.random.RandomState(seed)

        # Fix sigma so that SNR = s^2 / sigma^2 = 2.0 --> sigma = s / sqrt(2.0)
        if sigma is None:
            if gaussian:
                self.sigma = 0.2043
            else:
                self.sigma = 0.1840

        # Get data and targets
        if data is not None:
            self.data = data
            self.targets = targets

        else:
            data, targets = self._sample_dataset(self.n_samples, self.n_features, self.rho, self.sigma, self.gaussian)
            self.data, self.targets = torch.FloatTensor(data), torch.FloatTensor(targets)

            # z-score data and targets (normalization computed at SNR = 2.0)
            if normalize:
                if gaussian:
                    self.data_mu, self.data_sd = 0.0, 1.0
                    self.targets_mu, self.targets_sd = 0.03441, 0.3547

                else:  # uniform distribution
                    self.data_mu = 0.5 * np.sqrt(self.rho) + 0.5 * np.sqrt(1 - self.rho)
                    self.data_sd = 1 / np.sqrt(12)
                    self.targets_mu, self.targets_sd = 0.08565, 0.3187

                self.data = (self.data - self.data_mu) / self.data_sd
                self.targets = (self.targets - self.targets_mu) / self.targets_sd

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.shuffle_targets:
            y_index = self.rng.randint(len(self.targets))
        else:
            y_index = index
        return self.data[index], self.targets[y_index]

    def __len__(self):
        return len(self.data)

    def _sample_dataset(self, N, P, rho, sigma, gaussian):
        assert P >= 6, "n_features must be at least 6 for this dataset"

        # In the original paper X are sampled from a uniform distribution
        if gaussian:
            X = np.sqrt(rho) * self.rng.randn(N, 1) + np.sqrt(1 - rho) * self.rng.randn(N, P)
        else:
            X = np.sqrt(rho) * self.rng.rand(N, 1) + np.sqrt(1 - rho) * self.rng.rand(N, P)

        y = np.sin(X[:, 0] * (X[:, 0] + X[:, 1])) * np.cos(X[:, 2] + X[:, 3] * X[:, 4]) *\
            np.sin(np.exp(X[:, 4]) + np.exp(X[:, 5]) - X[:, 1])

        y += sigma * self.rng.randn(len(y))

        return X, y

    def get_feature_names(self):
        return self.features

    def get_groundtruth_features(self):
        return self.fea_groundtruth
