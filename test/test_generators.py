from __future__ import division
import unittest
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset
from torch.utils.data.dataloader import DataLoader
from modules.linregression import linreg_reconstruct

from utils import DDICT
from datasets.dataset_builder import build_dataset


args = DDICT(
    batch_size=128,
    lr=0.003,
    n_layers=4,
    n_features=10,
    n_hiddens=200,
    num_bins=200,
    p_dropout=0.5,
    epochs=100,
    n_samples=1000
)

def generate_correlated_data(args, corr=0.5):
    # Correlated Gaussians
    CC = (1 - corr) * torch.eye(args.n_features) + corr * torch.ones(args.n_features, args.n_features)

    gauss_data = torch.distributions.MultivariateNormal(torch.zeros(args.n_features), CC)
    tr_loader = DataLoader(TensorDataset(gauss_data.sample((args.n_samples,))), batch_size=args.batch_size, shuffle=False)

    gauss_data = torch.distributions.MultivariateNormal(torch.zeros(args.n_features), CC)
    te_loader = DataLoader(TensorDataset(gauss_data.sample((args.n_samples,))), batch_size=args.batch_size, shuffle=False)

    return tr_loader, te_loader


def compare_generator_to_linreg(gen, dataloader, idx_feature):
    gen.eval()

    with torch.no_grad():
        # True data
        X = dataloader.dataset[:][0]
        Xj = X[:, idx_feature]

        lin_reg = linreg_reconstruct(X, idx_feature)

        # Generator output
        out = gen.get_dataloader(dataloader, idx_feature).dataset[:][0]

        # check that non-generated features are the same
        idx = list(set(range(X.shape[1])) - set([idx_feature]))
        assert (X[:, idx] - out[:, idx]).abs().sum().item() == 0, "Non generated features are different"

        # Cut out predicted feature
        out_pred = out[:, idx_feature]
        err_gen = (out_pred - Xj).abs().mean().item()

        # Linear regression
        out_linreg = lin_reg(X[:, idx]).view(-1)
        err_linreg = (out_linreg - Xj).abs().mean().item()

    return err_gen, err_linreg


class test_generators(unittest.TestCase):

    def test_gerator_oracle(self):
        from modules.generators import generator_from_data

        torch.manual_seed(1)

        # options to pass to data builder
        data_opt = DDICT(
            dataset='sinexp',
            sinexp_gaussian=False,
            numSamples=500,
            Xdim=50,
            batchSize=100,
            num_workers=1,
            dataseed=0,
        )

        # rho = 1.0
        data_opt.sinexp_rho = 1.0
        dataloaders, _, _ = build_dataset(data_opt)
        dl_train_P, dl_train_Q, dl_test_P, dl_test_Q = dataloaders  # unpack

        gen, _ = generator_from_data(dl_train_P.dataset, 'oracle')

        gen_dl = gen.get_dataloader(dl_test_P, 0)

        self.assertAlmostEqual((gen_dl.dataset[:][0] - dl_test_P.dataset[:][0]).abs().mean().item(), 0.0, places=6)

        # rho = 0.5
        data_opt.sinexp_rho = 0.5
        dataloaders, _, _ = build_dataset(data_opt)
        dl_train_P, dl_train_Q, dl_test_P, dl_test_Q = dataloaders  # unpack

        gen, _ = generator_from_data(dl_train_P.dataset, 'oracle')

        gen_dl = gen.get_dataloader(dl_test_P, 0)

        self.assertEqual((gen_dl.dataset[:][0][:, 1:] - dl_test_P.dataset[:][0][:, 1:]).abs().mean().item(), 0.0)


    def test_gerator_classify(self):
        from modules.generators import GeneratorClassify
        from modules.generators import train_generator, test_generator

        tr_loader, te_loader = generate_correlated_data(args)
        gen_cl = GeneratorClassify(args.n_features, args.n_layers, args.n_hiddens, num_bins=args.num_bins, init_dataset=tr_loader, p_dropout=args.p_dropout)
        optimizer = optim.Adam(gen_cl.parameters(), lr=args.lr)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

        for epoch in range(args.epochs):
            tr_loss = train_generator(gen_cl, tr_loader, optimizer, log_times=0)
            te_loss = test_generator(gen_cl, nn.MSELoss(), te_loader)
            scheduler.step()
            print('{}: training loss = {:.3}\t test loss (mse) = {:.3}'.format(epoch, tr_loss, te_loss))

        for idx_feature in range(args.n_features):
            err_gen_cl, err_linreg = compare_generator_to_linreg(gen_cl, tr_loader, idx_feature)
            self.assertGreater(2.0 * err_linreg, err_gen_cl, 'Error of generator_classify is substantially higher than linear regression')


    def test_gerator_regress(self):
        from modules.generators import GeneratorRegress
        from modules.generators import train_generator, test_generator

        tr_loader, te_loader = generate_correlated_data(args)
        gen_reg = GeneratorRegress(args.n_features, args.n_layers, args.n_hiddens, p_dropout=args.p_dropout)
        optimizer = optim.Adam(gen_reg.parameters(), lr=args.lr)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

        for epoch in range(args.epochs):
            tr_loss = train_generator(gen_reg, tr_loader, optimizer, log_times=0)
            te_loss = test_generator(gen_reg, nn.MSELoss(), te_loader)
            scheduler.step()
            print('{}: training loss = {:.3}\t test loss (mse) = {:.3}'.format(epoch, tr_loss, te_loss))

        for idx_feature in range(args.n_features):
            err_gen_reg, err_linreg = compare_generator_to_linreg(gen_reg, tr_loader, idx_feature)
            self.assertGreater(2.0 * err_linreg, err_gen_reg, 'Error of generator_regress is substantially higher than linear regression')


if __name__ == '__main__':
    unittest.main()
