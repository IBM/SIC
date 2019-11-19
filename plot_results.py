from os import listdir
from os.path import isdir, isfile, join
from itertools import chain
import numpy as np
import matplotlib.pyplot as plt
from utils import shelf


def dlist(key, dat):
    r"""Runs over a list of dictionaries and outputs a list of values corresponding to `key`
        Short version (no checks): return np.array([d[key] for d in dat])
    """
    ret = []
    for i, d in enumerate(dat):
        if key in d:
            ret.append(d[key])
        else:
            print('key {} is not in dat[{}]. Skip.'.format(key, i))
    return np.array(ret)


def get_data(select_dict, ARGS, key_list, DAT):
    data = []
    for sel, key in zip(select_dict, key_list):
        # Select DAT
        k, v = next(iter(sel.items()))
        dat = [da[0] for da in zip(DAT, ARGS) if k in da[1] and da[1][k] == v][0]
        data.append(dlist(key, dat))
    return data


def color_bplot(bplot, colors):
    r"""Color the boxplots"""
    for patch, color in zip(bplot['boxes'], colors):
        patch.set_facecolor(color)
    for median in bplot['medians']:
        median.set(color='k', linewidth=1.5,)


def label_axis(ax, labels, xpos, ypos, fontsize=16, target_fdr=0.1):
    # Partially remove frame
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # y label
    ax.set_ylabel('Power and FDR', fontsize=fontsize)
    ax.set_ylim([-0.05, 1.05])

    # Hortizontal line for target fdr
    if target_fdr:
        ax.plot(ax.get_xlim(), [target_fdr, target_fdr], '--r')

    # New Axis
    new_ax = ax.twiny()
    new_ax.set_xticks(xpos)
    new_ax.set_xticklabels(labels)

    new_ax.xaxis.set_ticks_position('bottom')  # set the position of the second x-axis to bottom
    new_ax.xaxis.set_label_position('bottom')  # set the position of the second x-axis to bottom
    new_ax.spines['bottom'].set_position(('outward', ypos))  # positions below

    # Remove frame for new_ax
    new_ax.spines['bottom'].set_visible(False)
    new_ax.spines['top'].set_visible(False)
    new_ax.spines['left'].set_visible(False)
    new_ax.spines['right'].set_visible(False)

    new_ax.tick_params(length=0, labelsize=fontsize)
    new_ax.set_xlim(ax.get_xlim())

    return new_ax


if __name__ == "__main__":
    # Load data
    PATH = 'output/'
    DIRS = [d for d in listdir(PATH) if isdir(join(PATH, d))]
    FILES = [join(PATH, d, f) for d in DIRS for f in listdir(join(PATH, d))
             if isfile(join(PATH, d, f)) and f[-3:]=='.pt']

    ARGS, DAT, MODELS = [], [], []
    for f in FILES:
        sh = shelf()._load(f)
        ARGS.append(sh.args)
        if 'd' in sh:
            DAT.append(sh['d'])
            MODELS.append(sh.args['model'])
        else:
            print("WARNING: There is no data field d field in file {}. Skip.".format(f))
            continue

    # ---------------------------
    # Process data
    # ---------------------------
    select_dict, key_list, labels, positions, ax_labels, ax_positions = [], [], [], [-2], [], [-2]
    # Baseline models
    for m, l in zip(['en', 'rf'], ['Elastic Net', 'Random Forest']):
        if m in MODELS:
            select_dict += 4*[{'model': m}]
            key_list += ['tpr_selected', 'fdr_selected', 'hrt_tpr_selected', 'hrt_fdr_selected']
            labels += ['TPR', 'FDR', 'TPR\nHRT', 'FDR\nHRT']
            p = positions[-1] + 2
            positions += [1+p, 2+p,   4+p, 5+p]
            ax_labels += [l]
            ax_positions += [ax_positions[-1] + len(l)/2]

    # Our models
    for m, l, pos in zip(['sic_supervised', 'sic'], ['Sobolev Penalty', 'SIC'], [5.5, 4]):
        if m in MODELS:
            select_dict += 2*[{'model': m}]
            key_list += ['hrt_tpr_selected', 'hrt_fdr_selected']
            labels += ['TPR\nHRT', 'FDR\nHRT']
            p = positions[-1] + 2
            positions += [1+p, 2+p]
            ax_labels += [l]
            ax_positions += [ax_positions[-1] + pos]

    positions.pop(0);
    ax_positions.pop(0);

    data = get_data(select_dict, ARGS, key_list, DAT)

    # ---------------------------
    # Plot
    # ---------------------------
    dataset = ARGS[0]['dataset'].upper()
    n_samples = ARGS[0]['numSamples']

    fig = plt.figure(figsize=(8, 3))
    ax = plt.subplot(111)

    bplot = plt.boxplot(data, positions=positions, labels=labels, patch_artist=True)
    label_axis(ax, ax_labels, ax_positions, 32, fontsize=13)
    color_bplot(bplot, len(positions)//2*['lightblue', 'orange'])

    fig.suptitle(f'Dataset {dataset}, N={n_samples}');

    fig.tight_layout()
    fig.savefig(f"output/{dataset}_{n_samples}.png", bbox_inches='tight')
