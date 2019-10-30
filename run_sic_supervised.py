import argparse
import os, sys
import torch

from modules.models import init_D, init_optimizerD
from SIC_imports import compute_mse, eta_optim_step_, supervised_forward_sobolev_penalty
from utils import DDICT, shelf, avg_iterable
from datasets.dataset_builder import build_dataset

from stattests import hrt, compute_fdr, compute_tpr, log_metrics, log_metrics_selected

# configure event logging
import logging
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO, stream=sys.stdout)
logger = logging.getLogger('main')


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='sinexp', help='toy | bleitoy | liang | liang_switch | ccle | olfaction | biobank | hiv ')
parser.add_argument('--sinexp-rho', default=0.5, type=float, help='Correlation coefficient between pairs of covariates in SinExp dataset')
parser.add_argument('--sinexp-gaussian', action='store_true', help='Sample covariates from gaussian or uniform in SinExp dataset')
parser.add_argument('--data-seed', type=int, default=0, help='initial random seed for data')
parser.add_argument('--generator-type', default='classify', help='classify | regress')
parser.add_argument('--batchSize', type=int, default=50, help='input batch size')
parser.add_argument('--numSamples', type=int, default=250, help='(for toy) num Samples in train & heldout')
parser.add_argument('--Xdim', type=int, default=50,     help='(for toy) X dimensionality')
parser.add_argument('--layerSize', type=int, default=100, help='')
parser.add_argument('--wdecay', type=float, default=1e-3, help='')
parser.add_argument('--lrD', type=float, default=1e-3, help='learning rate for D = Sobolev Mut Info neural estimator')
parser.add_argument('--beta1', type=float, default=0.5, help='Adam optimizer for D: (beta1, beta2)')
parser.add_argument('--beta2', type=float, default=0.999, help='Adam optimizer for D: (beta1, beta2)')
parser.add_argument('--dropout', default=0.3, type=float, help='Discriminator/Critic dropout')
parser.add_argument('--lambdaSobolev', type=float, default=0.5, help='lambda on Sobolev constraint term')
parser.add_argument('--eta-lr', type=float, default=0.1, help='lr for eta; in case of L1^2 this is mirror descent scale')
parser.add_argument('--T', type=int, default=201, help='number of updates to D, training duration')
parser.add_argument('--log-every', type=int, default=10, help='interval to log, compute metrics on heldout')
parser.add_argument('--eta-step-type', default='mirror', help='mirror | reduced')
parser.add_argument('--seed', type=int, default=1238, help='random seed')
parser.add_argument('--ftdr-cutoff', default=6, type=int, help='fdr / tpr cutoff')
parser.add_argument('--do-hrt', action='store_true', help='perform HRT')
parser.add_argument('--target-fdr', default=0.1, type=float, help='target FDR for HRT')
parser.add_argument('--hrt-cutoff', type=int, default=20, help='maximal number of features for HRT to evaluate')
parser.add_argument('--n-critic', default=1, type=int, help='No. of critic updates before eta update')
parser.add_argument('--n-runs', type=int, default=100, help='number of repetitons over everything')
parser.add_argument('--nocuda', action='store_true', help='enables cuda')


def train_supervised(opt, dataloaders, net, groundtruth_feat=None, n_epochs=1, logger=None):
    """
        Args:
            opt (DDICT): parameters for training
            dataloaders (tuple): list of dataloaders
            net (nn.module): initialization for network `net`
    """
    dl_train_P, dl_train_Q, dl_test_P, dl_test_Q = dataloaders  # unpack

    # Optizer
    optimizerD = init_optimizerD(opt, net)

    # etas are initialized uniformly
    eta_x = torch.tensor([1 / opt.Xdim] * opt.Xdim, device=next(net.parameters()).device, requires_grad=True)

    if logger: logger.info(net)

    # Train
    if logger: logger.info('Start training')

    for epoch in range(n_epochs):
        for batch_idx, dataP in enumerate(dl_train_P):
            n_iter = epoch * len(dl_train_P) + batch_idx

            optimizerD.zero_grad()
            if hasattr(eta_x.grad, 'zero_'):
                eta_x.grad.zero_()

            mse_loss, constraint_Sobo = supervised_forward_sobolev_penalty(net, dataP[0], dataP[1], eta_x)

            obj_net = mse_loss + (opt.lambdaSobolev / 2) * constraint_Sobo

            obj_net.backward()
            optimizerD.step()

            if (n_iter + 1) % opt.n_critic == 0:
                eta_optim_step_(eta_x, opt.eta_step_type, opt.eta_lr)

        # eval / logging
        if logger and epoch % opt.log_every == 0:
            # Average test performance
            mse_loss_te, constraint_Sobo_te = avg_iterable(dl_test_P, lambda d: supervised_forward_sobolev_penalty(net, d[0], d[1], eta_x))

            msg = '[{:5d}]   TRAIN: mse={:.3f}, constr_Sobo={:.3f}   TEST: mse={:.3f}, constr_Sobo={:.3f}'\
                .format(epoch, mse_loss, constraint_Sobo, mse_loss_te, constraint_Sobo_te.item())

            # fdr and tpr
            if groundtruth_feat:
                _, eta_sortix = torch.sort(eta_x, descending=True)
                fdr = compute_fdr(eta_sortix.clone().detach().cpu(), groundtruth_feat, eta_sortix.size(0), cut_off=opt.ftdr_cutoff)
                tpr = compute_tpr(eta_sortix.clone().detach().cpu(), groundtruth_feat, eta_sortix.size(0), cut_off=opt.ftdr_cutoff)
                msg += '   FDR={:.3f}, TPR={:.3f}'.format(fdr, tpr)

            logger.info(msg)

    # End of training - saving and logging
    loss_tr = compute_mse(dl_train_P, net).item()
    loss_te = compute_mse(dl_test_P, net).item()

    if logger: logger.info('training MSE: {:.3f}, test MSE: {:.3f}'.format(loss_tr, loss_te))

    return net, eta_x, loss_tr, loss_te


# HRT risk function
def risk_model_fn(dataloader):
    global net
    return compute_mse(dataloader, net).item()


if __name__ == "__main__":
    args = parser.parse_args()

    args.model = 'sic_supervised'
    args.DiscArch = 'supervised_nobias'

    # Initialize input & output dirs
    args.outdir = os.path.join('output', 'sic_supervised')
    os.makedirs(args.outdir, exist_ok=True)

    # options to pass to data builder
    data_opt = DDICT(
        dataset=args.dataset,
        sinexp_gaussian=args.sinexp_gaussian,
        sinexp_rho=args.sinexp_rho,
        numSamples=args.numSamples,
        Xdim=args.Xdim,
        batchSize=args.batchSize,
        dataseed=args.data_seed,
    )

    if args.nocuda:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('device:', device)

    logger.info(str(args).replace(', ', '\n'))
    torch.manual_seed(args.seed)

    # Save everything in SHELF
    save_filename = os.path.join(args.outdir, 'mse_sobo') + '_' + args.dataset
    SH = shelf(args=args.__dict__)
    SH._save(save_filename, date=True)
    SH.d = []

    for n_iter in range(args.n_runs):
        logger.info("\n\n* Repetition {} of {}\n".format(n_iter + 1, args.n_runs))

        RES = dict(n_iter=n_iter)  # Results to save

        # Reload dataset (with fresh random seed)
        data_opt.dataseed += 1
        dataloaders, fea_names, groundtruth_feat = build_dataset(data_opt)

        logger.info(str(dataloaders[0].dataset))

        # Init and train model
        net = init_D(args, device)
        net, eta_x, loss_tr, loss_te = train_supervised(args, dataloaders, net, groundtruth_feat=groundtruth_feat, n_epochs=args.T, logger=logger)

        RES.update({'loss_tr': loss_tr, 'loss_te': loss_te})

        # metrics:
        eta_x = eta_x.detach()
        RES.update(log_metrics(eta_x, groundtruth_feat, args.ftdr_cutoff, key_prefix='', logger=logger))

        # HRT
        if args.do_hrt:
            hrt_sorted_selected_features, hrt_pvals = hrt(eta_x, risk_model_fn, dataloaders, hrt_cutoff=args.hrt_cutoff, target_fdr=args.target_fdr,
                                                          generator_type=args.generator_type, n_rounds=1000, logger=logger)
            RES.update({'hrt_pvals': hrt_pvals})
            RES.update(log_metrics_selected(hrt_sorted_selected_features, groundtruth_feat, key_prefix='hrt_', logger=logger))

        SH.d += [RES]
        SH._save()
