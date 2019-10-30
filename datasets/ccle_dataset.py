from __future__ import print_function
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets.utils import download_url
import os
import errno
import numpy as np
import pandas as pd


class CCLE_Dataset(torch.utils.data.Dataset):
    """`CCLE` dataset from paper:

    Barretina, J., Caponigro, G., Stransky, N., Venkatesan, K., Margolin, A. A., Kim, S., ... & Reddy, A. (2012).
     The Cancer Cell Line Encyclopedia enables predictive modelling of anticancer drug sensitivity.
     Nature, 483(7391), 603.

    Note:
        The X dataset is z-scored, which means that if it is partitioned into a training and test split, these have
        to be re-z-scored according to the training split

    Args:
        root (string): Root directory of dataset where ``processed/training.pt``
            and  ``processed/test.pt`` exist.
        task (string): Which task should provide output among:
            ['Bakery', 'Sour','Intensity','Sweet','Burnt','Pleasantness','Fish', 'Fruit','Garlic','Spices',
            'Cold','Acid','Warm', Musky','Sweaty','Ammonia','Decayed','Wood','Grass', 'Flower','Chemical']
        train (bool, optional): If True, creates dataset from the training split,
            otherwise from the test split. The two split come from the same partition if the
            random seed `seed` is the same.
        test_size (int, float): how much data has to be reserved for test.
            If test_size is int it will indicate the number of samples. If it's a float, it's the fraction
            of samples over the total.
        shuffle_target (bool): If True, it shuffle the targets (Y) compared to data (X) breaking the dependence between
            X and Y, such that P(X,Y) = P(X)P(Y). If False, X and Y are sampled together from P(X,Y).
        seed (int): seed of random number generator
        z_score (bool): whether to z-score X features or not. z-score statistics are always computed on the training split.
            Also, note that the whole X dataset is already z-score (see note above).
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
    """
    urls = [
        'https://www.dropbox.com/s/7iy0ght31hxhn7d/mutation.txt',
        'https://www.dropbox.com/s/bplwquwbc7zleck/expression.txt',
        'https://www.dropbox.com/s/78mp3ebnb4h6jsy/response.csv',
    ]
    download_option = '?dl=1'

    files = [
        'mutation.txt',
        'expression.txt',
        'response.csv'
    ]

    def __init__(self, root, task = 'PLX4720', feature_type ='both', train=True, test_size=0.1,
                 shuffle_targets=False, seed=1, z_score=True, download=True, verbose = False, parent_dataset = None):
        self.root = os.path.expanduser(root)

        if not isinstance(feature_type, str) or feature_type not in ['mutation', 'expression', 'both']:
            raise ValueError('task must be one of the following task descriptors: ' + str(['mutation', 'expression', 'both']))
        else:
            self.feature_type = feature_type

        self.fea_groundtruth = ['C11orf85', 'FXYD4', 'SLC28A2', 'MAML3_MUT', 'RAD51L1_MUT', 'GAPDHS', 'BRAF_MUT']

        self.task  = task # drug target

        self.train = train
        self.shuffle_targets = shuffle_targets
        self.verbose = verbose

        # Random number generator
        self.rng = np.random.RandomState(seed)

        if parent_dataset is None:
            if download:
                self.download()

            if not self._check_exists():
                raise RuntimeError('Dataset not found. You can use download=True to download it')

            self.full_data, self.full_targets, self.features, all_features = self.load_data()

            if isinstance(test_size, float):
                if test_size > 1.0 or test_size < 0.0:
                    raise ValueError('test_size must be integer or a float between 0.0 and 1.0')
                else:
                    self.test_size = int(len(self.full_data) * test_size)
            elif isinstance(test_size, int):
                if test_size >= len(self.full_data) or test_size < 0:
                    raise ValueError('integer test_size must be between 0 and {}'.format(len(self.full_data)))
                else:
                    self.test_size = test_size

            # Permutation indices:
            perm = self.rng.permutation(len(self.full_data))
            self.ind_train = perm[self.test_size:]
            self.ind_test  = perm[:self.test_size]

        else:
            self.full_data = parent_dataset.full_data
            self.full_targets = parent_dataset.full_targets
            self.features  = parent_dataset.features
            self.ind_train = parent_dataset.ind_train
            self.ind_test  = parent_dataset.ind_test

        # get feature indexes
        self.fea_groundtruth_idx = [self.features.get_loc(ftr) for ftr in self.fea_groundtruth]

        if self.train:
            self.data, self.targets = self.full_data[self.ind_train], self.full_targets[self.ind_train]
        else:
            self.data, self.targets = self.full_data[self.ind_test], self.full_targets[self.ind_test]

        # z-score according to training split
        if z_score:
            mu = np.mean(self.full_data[self.ind_train], 0)
            sd = np.std(self.full_data[self.ind_train], 0) + 1e-6
            self.data = (self.data - mu) / sd

            mu = np.mean(self.full_targets[self.ind_train], 0)
            sd = np.std(self.full_targets[self.ind_train], 0) + 1e-6
            self.targets = (self.targets - mu) / sd

        self.z_score = z_score

        self.data, self.targets = torch.FloatTensor(self.data), torch.FloatTensor(self.targets)


    def get_feature_names(self):
        return self.features.values

    def get_groundtruth_features(self):
        return self.fea_groundtruth_idx

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

    def _check_exists(self):
        return all(map(lambda f: os.path.exists(os.path.join(self.root, f)), self.files))

    def download(self):
        """Download the olfaction data if it doesn't exist in processed_folder already."""

        if self._check_exists():
            return

        # download files
        try:
            os.makedirs(os.path.join(self.root))
        except OSError as e:
            if e.errno == errno.EEXIST:
                pass
            else:
                raise

        for url in self.urls:
            filename = url.rpartition('/')[2]
            download_url(url + self.download_option, root=self.root, filename=filename, md5=None)

    def ccle_feature_filter(self, X, y, threshold=0.1):
        # Remove all features that do not have at least pearson correlation at threshold with y
        corrs = np.array([np.abs(np.corrcoef(x, y)[0,1]) if x.std() > 0 else 0 for x in X.T])
        selected = corrs >= threshold
        print(selected.sum(), selected.shape, corrs[34758])
        return selected, corrs

    def load_data(self):
        X_drugs, y_drugs, drugs, cells, features = self.load_ccle()
        drug_idx = drugs.get_loc(self.task)

        if self.verbose:
            print('Drug {}'.format(drugs[drug_idx]))

        X_drug, y_drug = X_drugs[drug_idx], y_drugs[drug_idx]

        # Specific to PLX4720. Filters out all features with pearson correlation less than 0.1 in magnitude
        if self.verbose:
            print('Filtering by correlation with signal first')
        ccle_selected, corrs = self.ccle_feature_filter(X_drug, y_drug)
        # keeps the ground truth features
        for plx4720_feat in self.fea_groundtruth:
            idx = features.get_loc(plx4720_feat)
            ccle_selected[idx] = True
            if self.verbose:
                print('Correlation for {}: {:.4f}'.format(plx4720_feat, corrs[idx]))
        ccle_features = features[ccle_selected]

        # uses data from filtered features only
        X_drug = X_drug[:, np.nonzero(ccle_selected)[0]]

        return X_drug, y_drug, ccle_features, features

    def load_ccle(self):
        r"""Load CCLE dataset
            This method is based on the code in https://github.com/tansey/hrt/blob/master/examples/ccle/main.py
            published together with the paper Tansey et al. (http://arxiv.org/abs/1811.00645)
            and is subject to the following license:

            The MIT License (MIT)

            Copyright (c) 2018 Wesley Tansey

            Permission is hereby granted, free of charge, to any person obtaining a copy
            of this software and associated documentation files (the "Software"), to deal
            in the Software without restriction, including without limitation the rights
            to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
            copies of the Software, and to permit persons to whom the Software is
            furnished to do so, subject to the following conditions:

            The above copyright notice and this permission notice shall be included in all
            copies or substantial portions of the Software.

            THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
            IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
            FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
            AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
            LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
            OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
            SOFTWARE.
        """

        if self.feature_type in ['expression', 'both']:
            # Load gene expression
            expression = pd.read_csv(os.path.join(self.root, self.files[1]), delimiter='\t', header=2, index_col=1).iloc[:,1:]
            expression.columns = [c.split(' (ACH')[0] for c in expression.columns]
            features = expression
        if self.feature_type in ['mutation', 'both']:
            # Load gene mutation
            mutations = pd.read_csv(os.path.join(self.root, self.files[0]), delimiter='\t', header=2, index_col=1).iloc[:,1:]
            mutations = mutations.iloc[[c.endswith('_MUT') for c in mutations.index]]
            features = mutations
        if self.feature_type == 'both':
            both_cells = set(expression.columns) & set(mutations.columns)
            z = {}
            for c in both_cells:
                exp = expression[c].values
                if len(exp.shape) > 1:
                    exp = exp[:,0]
                z[c] = np.concatenate([exp, mutations[c].values])
            both_df = pd.DataFrame(z, index=[c for c in expression.index] + [c for c in mutations.index])
            features = both_df
        response = pd.read_csv(os.path.join(self.root, self.files[2]), header=0, index_col=[0,2])

        # Get per-drug X and y regression targets
        cells = response.index.levels[0]
        drugs = response.index.levels[1]
        X_drugs = [[] for _ in drugs]
        y_drugs = [[] for _ in drugs]
        for j, drug in enumerate(drugs):
            if self.task is not None and drug != self.task:
                continue
            for i,cell in enumerate(cells):
                if cell not in features.columns or (cell, drug) not in response.index:
                    continue
                X_drugs[j].append(features[cell].values)
                y_drugs[j].append(response.loc[(cell,drug), 'Amax'])
            print('{}: {}'.format(drug, len(y_drugs[j])))

        X_drugs = [np.array(x_i) for x_i in X_drugs]
        y_drugs = [np.array(y_i) for y_i in y_drugs]

        return X_drugs, y_drugs, drugs, cells, features.index

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    CCLE Task: {}\n'.format(self.task)
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        tmp = 'train' if self.train is True else 'test'
        fmt_str += '    Split: {}\n'.format(tmp)
        fmt_str += '    Z-Score: {}\n'.format(self.z_score)
        fmt_str += '    Root Location: {}\n'.format(self.root)
        return fmt_str


if __name__=="__main__":

    DIR_DATASET = '~/data/ccle'

    # Common random seed to all datasets:
    random_seed = 123

    # P(X,X) distribution:
    trainset = CCLE_Dataset(DIR_DATASET, train = True)
    print (trainset)
    tr_P = DataLoader(trainset, batch_size=50, shuffle=True, num_workers=1)

    trainset_t = CCLE_Dataset(DIR_DATASET, train = False, parent_dataset = trainset)
    print (trainset_t)
    tr_P_t = DataLoader(trainset_t, batch_size=50, shuffle=True, num_workers=1)
