from torch.utils.data import DataLoader
import sys

from datasets.toy_dataset import ToyDataset, LiangSwitchingDataset, LiangDataset, SinExpDataset
from datasets.ccle_dataset import CCLE_Dataset


def build_dataset(opt):
    if opt.dataset in ['toy', 'liang', 'liang_switch']:
        # initalization of all toy datasets are the same
        DSCLASS = {'toy': ToyDataset,
                   'liang': LiangDataset, 'liang_switch': LiangSwitchingDataset}

        DSCLASS = DSCLASS[opt.dataset]

        ds_train_P = DSCLASS(opt.numSamples, opt.Xdim, seed=opt.dataseed)
        ds_train_Q = DSCLASS(opt.numSamples, opt.Xdim, seed=opt.dataseed,
                        betas=ds_train_P.betas,
                        data=ds_train_P.data, targets=ds_train_P.targets, shuffle_targets=True)

        ds_test_P = DSCLASS(opt.numSamples, opt.Xdim, seed=opt.dataseed+3,
                        betas=ds_train_P.betas)
        ds_test_Q = DSCLASS(opt.numSamples, opt.Xdim, seed=opt.dataseed+3,
                        betas=ds_train_P.betas,
                        data=ds_test_P.data, targets=ds_test_P.targets, shuffle_targets=True)

    elif opt.dataset == 'sinexp':
        ds_train_P = SinExpDataset(opt.numSamples, opt.Xdim, seed=opt.dataseed,
                                   rho = opt.sinexp_rho, gaussian = opt.sinexp_gaussian)
        ds_train_Q = SinExpDataset(opt.numSamples, opt.Xdim, seed=opt.dataseed,
                        betas=ds_train_P.betas, rho = opt.sinexp_rho, gaussian = opt.sinexp_gaussian,
                        data=ds_train_P.data, targets=ds_train_P.targets, shuffle_targets=True)

        ds_test_P = SinExpDataset(opt.numSamples, opt.Xdim, seed=opt.dataseed+3,
                        betas=ds_train_P.betas,  rho = opt.sinexp_rho, gaussian = opt.sinexp_gaussian)
        ds_test_Q = SinExpDataset(opt.numSamples, opt.Xdim, seed=opt.dataseed+3,
                        betas=ds_train_P.betas, rho = opt.sinexp_rho, gaussian = opt.sinexp_gaussian,
                        data=ds_test_P.data, targets=ds_test_P.targets, shuffle_targets=True)

    elif opt.dataset == 'ccle':
        ds_train_P = CCLE_Dataset(opt.dataroot, task=opt.task, train=True, test_size=opt.test_size, shuffle_targets=False,
                              seed=opt.dataseed, z_score=True, download=True)
        ds_train_Q = CCLE_Dataset(opt.dataroot, task=opt.task, train=True, test_size=opt.test_size, shuffle_targets=True,
                              seed=opt.dataseed, z_score=True, download=True, parent_dataset = ds_train_P)

        ds_test_P = CCLE_Dataset(opt.dataroot, task=opt.task, train=False, test_size=opt.test_size, shuffle_targets=False,
                          seed=opt.dataseed, z_score=True, download=True, parent_dataset = ds_train_P)
        ds_test_Q = CCLE_Dataset(opt.dataroot, task=opt.task, train=False, test_size=opt.test_size, shuffle_targets=True,
                              seed=opt.dataseed, z_score=True, download=True, parent_dataset = ds_train_P)
    else:
        raise ValueError('Please use one of the following for dataset: toy | ccle | olfaction.')

    tr_P = DataLoader(ds_train_P, batch_size=opt.batchSize, shuffle=True, drop_last = True)
    tr_Q = DataLoader(ds_train_Q, batch_size=opt.batchSize, shuffle=True, drop_last = True)

    # for test phase, one single batch is created: batch_size=np.inf
    te_P = DataLoader(ds_test_P, batch_size=sys.maxsize, shuffle=True)
    te_Q = DataLoader(ds_test_Q, batch_size=sys.maxsize, shuffle=True)

    # resets to the correct dimension
    opt.Xdim = ds_train_P.data.size(1)

    return [tr_P, tr_Q, te_P, te_Q], ds_train_P.get_feature_names(), ds_train_P.get_groundtruth_features()

