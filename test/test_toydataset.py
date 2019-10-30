from __future__ import division
import unittest
import torch
import numpy as np


class test_toydataset(unittest.TestCase):

    def test_sinexp(self):
        from datasets.toy_dataset import SinExpDataset

        torch.manual_seed(1)

        # Check consistency across random seeds
        ds1 = SinExpDataset(n_samples=1000, n_features=50, gaussian=False, seed=123)
        ds2 = SinExpDataset(n_samples=1000, n_features=50, gaussian=False, seed=123)

        self.assertAlmostEqual((ds1.data - ds2.data).abs().mean().item(), 0.0)
        self.assertAlmostEqual((ds1.targets - ds2.targets).abs().mean().item(), 0.0)

        # Check normalization for rho=0
        for gaussian in [True, False]:
            for n_features in [6, 20]:
                ds = SinExpDataset(n_samples=10000000, rho=0.0, n_features=n_features, gaussian=gaussian, seed=1)

                self.assertAlmostEqual(ds.data.mean().item(), 0.0, places=2)
                self.assertAlmostEqual(ds.data.std().item(), 1.0, places=2)

                self.assertAlmostEqual(ds.targets.mean().item(), 0.0, places=2)
                self.assertAlmostEqual(ds.targets.std().item(), 1.0, places=2)

        # Check normalization of only features for rho=0.5
        for gaussian in [True, False]:
            for n_features in [6, 20]:
                ds = SinExpDataset(n_samples=10000000, rho=0.0, n_features=n_features, gaussian=gaussian, seed=1)

                self.assertAlmostEqual(ds.data.mean().item(), 0.0, places=2)
                self.assertAlmostEqual(ds.data.std().item(), 1.0, places=2)

        # Check snr of y = f(X) relation
        def fn(X):
            y = np.sin(X[:, 0] * (X[:, 0] + X[:, 1])) * np.cos(X[:, 2] + X[:, 3] * X[:, 4]) *\
                np.sin(np.exp(X[:, 4]) + np.exp(X[:, 5]) - X[:, 1])
            return y

        for rho in [0.0, 0.5]:
            for n_features in [6, 20]:
                ds = SinExpDataset(n_samples=1000, rho=rho, n_features=n_features, gaussian=True, seed=1)

                X = ds.data * ds.data_sd + ds.data_mu
                y = ds.targets * ds.targets_sd + ds.targets_mu

                noise = ((y - fn(X))**2).mean().item()
                signal = (fn(X)**2).mean().item()

                self.assertGreater(signal, 1.5 * noise, msg="signal-to-noise smaller than 1.5, for Gaussian features and rho={}".format(rho))
                self.assertGreater(2.5 * noise, signal, msg="signal-to-noise greater than 2.5, for Gaussian features and rho={}".format(rho))

        for rho in [0.0, 0.5]:
            for n_features in [6, 20]:
                ds = SinExpDataset(n_samples=1000, rho=rho, n_features=n_features, gaussian=False, seed=1)

                X = ds.data * ds.data_sd + ds.data_mu
                y = ds.targets * ds.targets_sd + ds.targets_mu

                noise = ((y - fn(X))**2).mean().item()
                signal = (fn(X)**2).mean().item()

                self.assertGreater(signal, 1.0 * noise, msg="signal-to-noise smaller than 1.0, for Uniform features and rho={}".format(rho))
                self.assertGreater(3.0 * noise, signal, msg="signal-to-noise greater than 3.0, for Uniform features and rho={}".format(rho))


if __name__ == '__main__':
    unittest.main()
