import torch
import numpy as np
import torch.nn.functional as F
from modules.generators import generator_from_data
from sklearn.metrics import roc_auc_score
from utils import log_to_dict


def hrt_sobolev(etas, Ep_D, dataloaders, hrt_cutoff=None, target_fdr=0.1, generator_type='classify', n_rounds=1000, logger=None):
    r"""Performs Holdout Randomization Test from Tansey et al. (http://arxiv.org/abs/1811.00645)
        using Sobolev distance as risk model

        Args:
            etas (torch.Tensor): sequence of etas
            Ep_D (functin): integration of discriminator
            dataloaders (list): list of dataloaders
            hrt_cutoff (int): number of top etas to consider
    """
    dl_train_P, dl_train_Q, dl_test_P, dl_test_Q = dataloaders  # unpack

    _, eta_sortix = torch.sort(etas, descending=True)

    # Only consider top `hrt_cutoff` features
    if hrt_cutoff is not None:
        eta_sortix = eta_sortix[:hrt_cutoff]

    # Compute first term of Sobolev distance (notice the '-', since it's a sup)
    t_h = -Ep_D(dl_test_P)

    # Instantiate generator
    if logger: logger.info('HRT: training generator')
    generator, _ = generator_from_data(dl_train_P.dataset, generator_type)

    # Loop over features
    p_vals = []
    if logger: logger.info('HRT: analyzing {} features'.format(len(eta_sortix)))

    for i, j in enumerate(eta_sortix):
        if logger: logger.info('   HRT testing feature {}\t({} of {})'.format(j, i + 1, len(eta_sortix)))
        t_j = []
        gen_j_data = generator.get_dataloader(dl_test_P, j)
        for r in range(n_rounds):
            # Sample from P_{j|-j}
            gen_j_data.dataset.resample_replaced_feature()
            # Compute empirical risk (notice the '-')
            t_j.append(-Ep_D(gen_j_data))
        p_vals.append(get_pval(t_h, t_j))
        if logger: logger.info('        eta={:.3f} (p={:.3f})'.format(etas[j], p_vals[-1]))

    # Use Benjamini-Hochberg to calibrate FDR
    idx_sorted_selected = bh(p_vals, target_fdr)
    sorted_selected_features = np.array([eta_sortix[j].item() for j in idx_sorted_selected])

    return sorted_selected_features, np.array(p_vals)


def hrt(etas, risk_model, dataloaders, hrt_cutoff=None, target_fdr=0.1, generator_type='regress', n_rounds=1000, logger=None):
    r"""Performs Holdout Randomization Test from Tansey et al. (http://arxiv.org/abs/1811.00645)

        Args:
            etas (torch.Tensor): etas to test
            risk_model (function): returns the risk of the model given a dataloader
            dataloaders (list): list of dataloaders

        Returns:
    """
    dl_train_P, dl_train_Q, dl_test_P, dl_test_Q = dataloaders  # unpack

    eta_sortix = torch.argsort(etas, descending=True)
    if hrt_cutoff is None:
        hrt_cutoff = len(etas)

    # Cut off all etas that are zero by setting cut_off at fist zero eta
    if etas[eta_sortix[hrt_cutoff]] == 0.0:
        hrt_cutoff = torch.nonzero(etas[eta_sortix] <= 0.0)[0].item()

    eta_sortix = eta_sortix[:hrt_cutoff]

    # Compute risk
    t_h = risk_model(dl_test_P)

    # Instantiate generator
    if logger: logger.info('HRT: training generator')
    generator, _ = generator_from_data(dl_train_P.dataset, generator_type)

    # Loop over features
    p_vals = []
    if logger: logger.info('HRT: analyzing {} features'.format(len(eta_sortix)))

    for i, j in enumerate(eta_sortix):
        if logger: logger.info('   HRT testing feature {}\t({} of {})'.format(j, i + 1, len(eta_sortix)))
        t_j = []
        gen_j_data = generator.get_dataloader(dl_test_P, j)
        for r in range(n_rounds):
            # Sample from P_{j|-j}
            gen_j_data.dataset.resample_replaced_feature()
            # Compute empirical risk
            t_j.append(risk_model(gen_j_data))
        p_vals.append(get_pval(t_h, t_j))
        if logger: logger.info('        eta={:.3f} (p={:.3f})'.format(etas[j], p_vals[-1]))

    # Use Benjamini-Hochberg to calibrate FDR
    idx_sorted_selected = bh(p_vals, target_fdr)
    sorted_selected_features = np.array([eta_sortix[j].item() for j in idx_sorted_selected])

    return sorted_selected_features, np.array(p_vals)


def compute_fdr(eta_sortix, ground_truth, total_features, axis=None, cut_off=40):
    r"""Computes FDR
    """
    truth = np.zeros(total_features)
    pred = np.zeros(total_features)

    assert len(ground_truth)<=cut_off, "Cut-off feature length more than ground_truth features "

    eta_sortix = eta_sortix[:cut_off]
    truth[ground_truth] = 1
    pred[eta_sortix] = 1

    return ((pred==1) & (truth==0)).sum(axis=axis) / pred.sum(axis=axis).astype(float).clip(1,np.inf)


def compute_tpr(eta_sortix, ground_truth, total_features, axis=None, cut_off=40):
    r"""Computes TPR
    """
    truth = np.zeros(total_features)
    pred = np.zeros(total_features)

    assert len(ground_truth)<=cut_off, "Cut-off feature length more than ground_truth features "

    eta_sortix = eta_sortix[:cut_off]
    truth[ground_truth] = 1
    pred[eta_sortix] = 1

    return ((pred==1) & (truth==1)).sum(axis=axis) / truth.sum(axis=axis).astype(float).clip(1,np.inf)



def get_pval(t, t_list, invert=False):
    r"""Computes p-values by counting how often elements of `t_list` are below `t` (or above, if `invert` is True)

        Args:
            t (float): statistics
            t_list (list): samples from null-distribution
            invert (bool): whether to count t >= t_list or t <= t_list
                `invert` should be `False` if `t` is a cost, and should be `True` if it is an importance weight
    """
    K = len(t_list)
    if invert:
        total = 1 + (t <= np.array(t_list)).sum()
    else:
        total = 1 + (t >= np.array(t_list)).sum()
    return total / (K + 1)


def auc_score(scores, ground_truth):
    r"""Computes the roc auc score from scores and a list of the ground truth discoveries
    """
    y_scores = np.array(scores).reshape(-1)
    y_true = np.zeros(len(y_scores))
    y_true[ground_truth] = 1
    auc = roc_auc_score(y_true, y_scores)
    return auc


def fdr_tpr(selected_features, ground_truth):
    r"""Computes the FDR and TPR @ discoveries number
    """
    n_discoveries = len(selected_features)
    P = len(ground_truth)
    TP = len(set(selected_features).intersection(set(ground_truth)))
    tpr = TP / P
    fdr = len(set(selected_features) - set(ground_truth)) / n_discoveries
    return fdr, tpr


def auc_fdr_tpr_curves(scores, ground_truth):
    if isinstance(scores, torch.Tensor):
        scores = np.array(scores.cpu().detach(), dtype=np.float)
    scores = scores.reshape(-1)

    fdr_curve, tpr_curve = np.zeros(len(scores)), np.zeros(len(scores))
    scores_sortix = np.argsort(-scores)
    for i in range(len(scores_sortix)):
        fdr_curve[i], tpr_curve[i] = fdr_tpr(scores_sortix[:i + 1], ground_truth)
    auc = auc_score(scores, ground_truth)
    return auc, fdr_curve, tpr_curve


def selected_fdr_tpr_curves(sorted_selected_features, ground_truth):
    r"""Takes a list of sorted selected features and the ground truth dicoveries, and build fdr and tpr curves

        Args:
            sorted_selected_features (list): list of selected features sorted from higher to lower importance
            ground_truth (list): list of groud truth dicoveries
    """
    sorted_selected_features = np.array(sorted_selected_features)

    fdr_curve, tpr_curve = np.zeros(len(sorted_selected_features)), np.zeros(len(sorted_selected_features))
    for i in range(len(fdr_curve)):
        fdr_curve[i], tpr_curve[i] = fdr_tpr(sorted_selected_features[:i + 1], ground_truth)
    return fdr_curve, tpr_curve


def bh(p, fdr):
    r"""Performs Benjamini-hochberg
    """
    p_orders = np.argsort(p)
    discoveries = []
    m = float(len(p_orders))
    for k, s in enumerate(p_orders):
        if p[s] <= (k + 1) / m * fdr:
            discoveries.append(s)
        else:
            break
    return np.array(discoveries)


def log_metrics(etas, fea_groundtruth, ftdr_cutoff, key_prefix='', logger=None):
    if isinstance(etas, torch.Tensor):
        etas = np.array(etas.cpu().detach(), dtype=np.float)
    etas = etas.reshape(-1)

    _, fdr_curve, tpr_curve = auc_fdr_tpr_curves(etas, fea_groundtruth)
    fdr_at, tpr_at = fdr_curve[ftdr_cutoff - 1], tpr_curve[ftdr_cutoff - 1]

    eta_sortix = np.argsort(-etas)
    selected_features = eta_sortix[np.nonzero(etas[eta_sortix] > 0)[0]]
    if logger:
        logger.info(" {} selected features {}: {}".format(key_prefix, len(selected_features), selected_features))
        logger.info(" {}FDR @ {}: {:.3f}".format(key_prefix, ftdr_cutoff, fdr_at))
        logger.info(" {}TPR @ {}: {:.3f}".format(key_prefix, ftdr_cutoff, tpr_at))

    if len(selected_features) > 0:
        fdr_selected = fdr_curve[len(selected_features) - 1]
        tpr_selected = tpr_curve[len(selected_features) - 1]
    else:
        fdr_selected, tpr_selected = 0.0, 0.0

    if logger:
        logger.info(" {}FDR @ selected: {:.3f}".format(key_prefix, fdr_selected))
        logger.info(" {}TPR @ selected: {:.3f}".format(key_prefix, tpr_selected))

    return log_to_dict(['selected_features', 'fdr_curve', 'tpr_curve', 'fdr_at', 'tpr_at', 'fdr_selected', 'tpr_selected'],
                       locals(), key_prefix=key_prefix)


def log_metrics_selected(selected_features, fea_groundtruth, key_prefix='', logger=None):
    r"""FDR and TPR are computed at len(sorted_selected_features)
    """
    if hasattr(selected_features, 'reshape'):
        selected_features = selected_features.reshape(-1).tolist()

    n_features = len(selected_features)

    fdr_curve, tpr_curve = selected_fdr_tpr_curves(selected_features, fea_groundtruth)

    if n_features > 0:
        fdr_selected, tpr_selected = fdr_curve[-1], tpr_curve[-1]
    else:
        fdr_selected, tpr_selected = 0.0, 0.0

    if logger:
        logger.info(" {} selected features {}: {}".format(key_prefix, n_features, selected_features))
        logger.info(" {}FDR @ selected: {:.3f}".format(key_prefix, fdr_selected))
        logger.info(" {}TPR @ selected: {:.3f}".format(key_prefix, tpr_selected))

    return log_to_dict(['selected_features', 'fdr_curve', 'tpr_curve', 'fdr_selected', 'tpr_selected'],
                       locals(), key_prefix=key_prefix)
