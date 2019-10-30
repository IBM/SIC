import argparse
import os, sys
import torch
import numpy as np

from utils import DDICT, shelf
from stattests import hrt, log_metrics, log_metrics_selected
from datasets.dataset_builder import build_dataset

from sklearn.linear_model import ElasticNetCV
from sklearn.ensemble import RandomForestRegressor

# configure event logging
import logging
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO, stream=sys.stdout)
logger = logging.getLogger('main')


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='sinexp', help='toy | sinexp | liang | liang_switch | ccle | olfaction | biobank | hiv ')
parser.add_argument('--sinexp-rho', default=0.5, type=float, help='Correlation coefficient between pairs of covariates in SinExp dataset')
parser.add_argument('--sinexp-gaussian', action='store_true', help='Sample covariates from gaussian or uniform in SinExp dataset')
parser.add_argument('--generator-type', default='classify', help='classify | regress')
parser.add_argument('--numSamples', type=int, default=250, help='(for toy) num Samples in train & heldout')
parser.add_argument('--Xdim', type=int, default=50, help='(for toy) X dimensionality')
parser.add_argument('--ftdr-cutoff', default=6, type=int, help='fdr / tpr cutoff')
parser.add_argument('--model', default='en', help='en (elastic net) | rf (random forest)')
parser.add_argument('--do-hrt', action='store_true', help='perform HRT')
parser.add_argument('--hrt-cutoff', type=int, default=20, help='maximal number of features for HRT to evaluate')
parser.add_argument('--target-fdr', default=0.1, type=float, help='target FDR for HRT')
parser.add_argument('--n-runs', type=int, default=100, help='number of repetitons over everything')
parser.add_argument('--data-seed', type=int, default=0, help='initial random seed for data')


def train_sklearn_model(dataloaders, sklearn_model=ElasticNetCV, init_kwargs={'cv': 5}, importance_attr='coef_', logger=True):
    dl_train_P, dl_train_Q, dl_test_P, dl_test_Q = dataloaders  # unpack
    X, y = dl_train_P.dataset[:]
    y = y.view(-1)
    y = np.array(y.view(-1), dtype=np.float)

    model = sklearn_model(**init_kwargs)
    model_name = model.__repr__().split('(')[0]

    if logger: logger.info('Start training {}'.format(model_name))

    model.fit(X, y)

    # heldout eval / printing
    X_te, y_te = dl_test_P.dataset[:]
    y_te = y_te.view(-1)
    y_te = np.array(y_te.view(-1), dtype=np.float)

    loss_tr = ((model.predict(X) - y)**2).mean()
    loss_te = ((model.predict(X_te) - y_te)**2).mean()

    if logger: logger.info('{}, training MSE: {:.3f}, test MSE: {:.3f}'.format(model_name, loss_tr, loss_te))

    etas = torch.FloatTensor(np.abs(getattr(model, importance_attr)))
    return model, etas, loss_tr, loss_te


# HRT risk function
def risk_model_fn(dataloader):
    global model
    X, y = dataloader.dataset[:]
    X = np.array(X, dtype=np.float)
    y = np.array(y.view(-1), dtype=np.float)
    mse_loss = ((model.predict(X) - y)**2).mean()
    return mse_loss


if __name__ == "__main__":
    args = parser.parse_args()

    # Initialize input & output dirs
    args.outdir = os.path.join('output', 'baselines')
    os.makedirs(args.outdir, exist_ok=True)
    logger.info(str(args).replace(', ', '\n'))

    # options to pass to data builder
    data_opt = DDICT(
        dataset=args.dataset,
        sinexp_gaussian=args.sinexp_gaussian,
        sinexp_rho=args.sinexp_rho,
        numSamples=args.numSamples,
        Xdim=args.Xdim,
        batchSize=100,
        num_workers=1,
        dataseed=args.data_seed,
    )

    # sklearn model
    if args.model == 'en':
        sk_model = ElasticNetCV
        sk_init_kwargs = {'cv': 5, 'n_jobs': 1}
        sk_importance_attr = 'coef_'

    elif args.model == 'rf':
        sk_model = RandomForestRegressor
        sk_init_kwargs = {'n_estimators': 10, 'n_jobs': 1}
        sk_importance_attr = 'feature_importances_'

    else:
        raise ValueError("Sklearn model {} not recognized".format(args.model))

    # Save everything in SHELF
    save_filename = os.path.join(args.outdir, args.model) + '_' + args.dataset
    SH = shelf(args=args.__dict__)
    SH._save(save_filename, date=True)
    SH.d = []

    for n_iter in range(args.n_runs):
        logger.info("\n\n* Repetition {} of {}\n".format(n_iter + 1, args.n_runs))

        RES = dict(n_iter=n_iter)  # Results to save

        # Reload dataset (with fresh random seed)
        data_opt.dataseed += 1
        dataloaders, fea_names, fea_groundtruth = build_dataset(data_opt)

        logger.info(str(dataloaders[0].dataset))

        # Train model
        model, eta_x, loss_tr, loss_te = train_sklearn_model(dataloaders, sk_model, sk_init_kwargs, sk_importance_attr, logger=logger)
        RES.update({'loss_tr': loss_tr, 'loss_te': loss_te})

        # metrics:
        RES.update(log_metrics(eta_x, fea_groundtruth, args.ftdr_cutoff, key_prefix='', logger=logger))

        # HRT
        if args.do_hrt:
            hrt_sorted_selected_features, hrt_pvals = hrt(eta_x, risk_model_fn, dataloaders, hrt_cutoff=args.hrt_cutoff, target_fdr=args.target_fdr,
                                                          generator_type=args.generator_type, n_rounds=1000, logger=logger)
            RES.update({'hrt_pvals': hrt_pvals})
            RES.update(log_metrics_selected(hrt_sorted_selected_features, fea_groundtruth, key_prefix='hrt_', logger=logger))

        SH.d += [RES]
        SH._save()
