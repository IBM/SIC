import os
import time
from random import randint
from functools import reduce
import torch
from torch import nn
import numpy as np


def describe(t):
    """Returns a string describing an array
        Args
            t (numpy.array or torch.tensor): array of data

        Returns
            string describing array t
    """
    t = t.data if isinstance(t, torch.Tensor) else t
    s = '{:8s} [{:.4f} , {:.4f}] m+-s = {:.4f} +- {:.4f}'
    si = 'x'.join(map(str, t.shape if isinstance(t, np.ndarray) else t.size()))
    return s.format(si, t.min(), t.max(), t.mean(), t.std())


def log_current_variables(tbw, n_iter, all_data, keys_to_log, key_prefix='', tb_trunc_tensor_size=10):
    for k in keys_to_log:
        v = all_data[k]
        logkey = key_prefix + k
        if isinstance(v, torch.Tensor):
            v = v.detach().cpu()  # get out of autograd
            if v.numel() == 1:
                tbw.add_scalar(logkey, v.item(), n_iter)
            elif v.dim() == 1 and v.numel() <= tb_trunc_tensor_size:
                # log as scalar group [0, D[. make dictionary.
                v = {str(i): v[i].item() for i in range(len(v))}
                tbw.add_scalars(logkey, v, n_iter)
            else:
                vtrunc = {str(i): v[i].item() for i in range(tb_trunc_tensor_size)}
                tbw.add_scalars(logkey, vtrunc, n_iter)
                tbw.add_histogram(logkey, v.numpy(), n_iter)
        else:
            tbw.add_scalar(logkey, v, n_iter)


class DDICT:
    """DotDictionary, dictionary whose items can be accesses with the dot operator

        E.g.
        >> args = DDICT(batch_size=128, epochs=10)
        >> print(args.batch_size)
    """
    def __init__(self, **kwds):
        self.__dict__.update(kwds)

    def __repr__(self):
        return str(self.__dict__)

    def __iter__(self):
        return self.__dict__.__iter__()

    def __len__(self):
        return len(self.__dict__)

    def __setitem__(self, key, value):
        self.__dict__[key] = value

    def __getitem__(self, key):
        return self.__dict__[key]


def get_devices(cuda_device="cuda:0", seed=1):
    """Gets cuda devices
    """
    device = torch.device(cuda_device)
    torch.manual_seed(seed)
    # Multi GPU?
    num_gpus = torch.cuda.device_count()
    if device.type != 'cpu':
        print('\033[93m' + 'Using CUDA,', num_gpus, 'GPUs\033[0m')
        torch.cuda.manual_seed(seed)
    return device, num_gpus


def make_data_parallel(module, expose_methods=None):
    """Wraps `nn.Module object` into `nn.DataParallel` and links methods whose name is listed in `expose_methods`
    """
    dp_module = nn.DataParallel(module)

    if expose_methods is None:
        if hasattr(module, 'expose_methods'):
            expose_methods = module.expose_methods

    if expose_methods is not None:
        for mt in expose_methods:
            setattr(dp_module, mt, getattr(dp_module.module, mt))
    return dp_module


class shelf(object):
    '''Shelf to save stuff to disk. Basically a DDICT which can save to disk.

    Example:
        SH = shelf(lr=[0.1, 0.2], n_hiddens=[100, 500, 1000], n_layers=2)
        SH._extend(['lr', 'n_hiddens'], [[0.3, 0.4], [2000]])
        # Save to file:
        SH._save('my_file', date=False)
        # Load shelf from file:
        new_dd = shelf()._load('my_file')
    '''
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def __add__(self, other):
        if isinstance(other, type(self)):
            sum_dct = copy.copy(self.__dict__)
            for k, v in other.__dict__.items():
                if k not in sum_dct:
                    sum_dct[k] = v
                else:
                    if type(v) is list and type(sum_dct[k]) is list:
                        sum_dct[k] = sum_dct[k] + v
                    elif type(v) is not list and type(sum_dct[k]) is list:
                        sum_dct[k] = sum_dct[k] + [v]
                    elif type(v) is list and type(sum_dct[k]) is not list:
                        sum_dct[k] = [sum_dct[k]] + v
                    else:
                        sum_dct[k] = [sum_dct[k]] + [v]
            return shelf(**sum_dct)

        elif isinstance(other, dict):
            return self.__add__(shelf(**other))
        else:
            raise ValueError("shelf or dict is required")

    def __radd__(self, other):
        return self.__add__(other)

    def __repr__(self):
        items = ("{}={!r}".format(k, self.__dict__[k]) for k in self._keys())
        return "{}({})".format(type(self).__name__, ", ".join(items))

    def __eq__(self, other):
        return self.__dict__ == other.__dict__

    def __iter__(self):
        return self.__dict__.__iter__()

    def __len__(self):
        return len(self.__dict__)

    def __setitem__(self, key, value):
        self.__dict__[key] = value

    def __getitem__(self, key):
        return self.__dict__[key]

    @staticmethod
    def _flatten_dict(d, parent_key='', sep='_'):
        "Recursively flattens nested dicts"
        items = []
        for k, v in d.items():
            new_key = parent_key + sep + k if parent_key else k
            if isinstance(v, MutableMapping):
                items.extend(shelf._flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)

    def _extend(self, keys, values_list):
        if type(keys) not in (tuple, list):  # Individual key
            if keys not in self._keys():
                self[keys] = values_list
            else:
                self[keys] += values_list
        else:
            for key, val in zip(keys, values_list):
                if type(val) is list:
                    self._extend(key, val)
                else:
                    self._extend(key, [val])
        return self

    def _keys(self):
        return tuple(sorted([k for k in self.__dict__ if not k.startswith('_')]))

    def _values(self):
        return tuple([self.__dict__[k] for k in self._keys()])

    def _items(self):
        return tuple(zip(self._keys(), self._values()))

    def _save(self, filename=None, date=True):
        if filename is None:
            if not hasattr(self, '_filename'):  # First save
                raise ValueError("filename must be provided the first time you call _save()")
            else:  # Already saved
                torch.save(self, self._filename + '.pt')
        else:  # New filename
            if date:
                filename += '_' + time.strftime("%Y%m%d-%H:%M:%S")
            # Check if filename does not already exist. If it does, change name.
            while os.path.exists(filename + '.pt') and len(filename) < 100:
                filename += str(randint(0, 9))
            self._filename = filename
            torch.save(self, self._filename + '.pt')
        return self

    def _load(self, filename):
        try:
            self = torch.load(filename)
        except FileNotFoundError:
            self = torch.load(filename + '.pt')
        return self

    def _to_dict(self):
        "Returns a dict (it's recursive)"
        return_dict = {}
        for k, v in self.__dict__.items():
            if isinstance(v, type(self)):
                return_dict[k] = v._to_dict()
            else:
                return_dict[k] = v
        return return_dict

    def _flatten(self, parent_key='', sep='_'):
        "Recursively flattens nested ddicts"
        d = self._to_dict()
        return shelf._flatten_dict(d)


def log_to_dict(keys_to_log, scope, key_prefix=''):
    """
    Examples::
        >>> a,b = 1.0, 2.0
        >>> d = log_to_dict(['a', 'b'], d, locals())
        >>> d
        >>>     {'a': 1.0, 'b': 2.0}
    """
    d = dict()
    for k in keys_to_log:
        v = scope[k]
        if isinstance(v, torch.Tensor):
            v = v.detach().cpu()  # get out of autograd
            v = np.array(v, dtype=np.float)
        d[key_prefix + k] = v
    return d


def avg_iterable(iterable, func):
    '''Applies function `func` to each element of `iterable` and averages the results

        Args:
            iterable: an iterable
            func: function being applied on each element of `iterable`

        Returns:
            Average of `func` applied on `iterable`
    '''
    lst = [func(it) for it in iterable]
    return [sum(x) / len(lst) for x in zip(*lst)]
