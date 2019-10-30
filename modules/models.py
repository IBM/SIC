import torch
import torch.nn as nn


def add_layer(seq, ix, n_inputs, n_outputs, nonlin, normalization):
    seq.add_module('L'+str(ix), nn.Linear(n_inputs, n_outputs))
    if ix > 0 and normalization: # only LN/IN after first layer.
        if normalization == 'LN':
            seq.main.add_module('A'+str(ix), nn.LayerNorm(n_outputs))
        else:
            raise ValueError('Unknown normalization: {}'.format(normalization))
    if nonlin == 'LeakyReLU':
        seq.add_module('N'+str(ix), nn.LeakyReLU(0.2, inplace=True))
    elif nonlin == 'ReLU':
        seq.add_module('N'+str(ix), nn.ReLU(inplace=True))
    elif nonlin == 'Sigmoid':
        seq.add_module('N'+str(ix), nn.Sigmoid())


class D_phiVpsi(nn.Module):
    def __init__(self, insizes=[1,1], layerSizes=[ [32,32,16] ]*2, nonlin='LeakyReLU', normalization=None):
        super(D_phiVpsi, self).__init__()
        self.phi_x, self.psi_y = nn.Sequential(), nn.Sequential()
        # phi_x and psi_y same arch (by layerSizes)
        for seq, insize, layerSize in [(self.phi_x, insizes[0], layerSizes[0]), (self.psi_y, insizes[1], layerSizes[1])]:
            for ix, n_inputs, n_outputs in zip(range(len(layerSize)), [insize]+layerSize[:-1], layerSize):
                add_layer(seq, ix, n_inputs, n_outputs, nonlin, normalization)
        self.phiD, self.psiD = layerSizes[0][-1], layerSizes[1][-1]
        # inner matrix in bilinear form
        self.W = nn.Parameter(torch.randn(self.phiD, self.psiD))

    def forward(self, x, y):
        x = x.view(x.size(0), -1) # bs x D with D >=1
        y = y.view(x.size(0), 1)  # bs x 1
        phi_x = self.phi_x(x)
        psi_y = self.psi_y(y)
        out = (torch.mm(phi_x, self.W) * psi_y).sum(1, keepdim=True)
        return out

class D_concat(nn.Module):
    def __init__(self, insizes=[1,1], layerSizes=[32,32,16], nonlin='LeakyReLU', normalization=None):
        super(D_concat, self).__init__()
        insize = sum(insizes)
        self.main = nn.Sequential()
        for ix, n_inputs, n_outputs in zip(range(len(layerSizes)), [insize]+layerSizes[:-1], layerSizes):
            add_layer(self.main, ix, n_inputs, n_outputs, nonlin, normalization)
            self.PhiD = n_outputs
        self.V = nn.Linear(self.PhiD, 1, bias=False)
        self.V.weight.data *= 100
    def forward(self, x, y):
        x = x.view(x.size(0), -1) # bs x D with D >=1
        y = y.view(x.size(0), 1)  # bs x 1
        inp = torch.cat( [x,y], dim=1)
        phi = self.main(inp)
        return self.V(phi)


class D_concat2(nn.Module):
    def __init__(self, insizes=[1,1], layerSize=100):
        super(D_concat2, self).__init__()
        self.branchx = nn.Sequential(
            nn.Linear(insizes[0], layerSize),
            nn.LeakyReLU(),
            nn.Linear(layerSize, layerSize),
            nn.LeakyReLU(),
        )
        self.branchy = nn.Sequential(
            nn.Linear(insizes[1], layerSize),
            nn.LeakyReLU(),
            nn.Linear(layerSize, layerSize),
            nn.LeakyReLU(),
        )
        self.branchxy = nn.Sequential(
            nn.Linear(2*layerSize, layerSize),
            nn.LeakyReLU(),
            nn.Linear(layerSize, layerSize),
            nn.LeakyReLU(),
            nn.Linear(layerSize, 1),
        )
    def forward(self, x, y):
        x = x.view(x.size(0), -1) # bs x D with D >=1
        y = y.view(x.size(0), 1)  # bs x 1
        xy = torch.cat([self.branchx(x), self.branchy(y)], dim=1)
        return self.branchxy(xy)


class D_concat_first(nn.Module):
    def __init__(self, insize=2, layerSize=100, dropout=0.0):
        super(D_concat_first, self).__init__()
        self.branchxy = nn.Sequential(
            nn.Linear(insize, layerSize, bias=False),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(layerSize, layerSize, bias=False),
            nn.ReLU(),
            nn.Dropout(p=dropout)
            )
        self.last_linear = nn.Linear(layerSize, 1, bias=False)
    def forward(self, x, y):
        x = x.view(x.size(0), -1)
        y = y.view(x.size(0), 1)

        xy = torch.cat([x,y], dim=1)
        return self.last_linear(self.branchxy(xy))


class D_supervised_nobias(nn.Module):
    def __init__(self, n_inputs, n_outputs, layerSize=100, dropout=0.0, bias=False):
        super().__init__()

        self.n_inputs = n_inputs
        self.net = nn.Sequential(
            nn.Linear(n_inputs, layerSize, bias=bias),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(layerSize, layerSize, bias=bias),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(layerSize, 1, bias=bias)
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.net(x)


class D_supervised(nn.Module):
    def __init__(self, n_inputs, n_outputs, layerSize=100, dropout=0.0, bias=True):
        super(D_supervised, self).__init__()

        self.n_inputs = n_inputs
        self.net = nn.Sequential(
            nn.Linear(n_inputs, layerSize, bias=bias),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(layerSize, layerSize, bias=bias),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(layerSize, 1, bias=bias)
        )
        self.mask = None # to set after creation

    def set_mask(self, mask):
        assert mask.dim() == 1 and mask.size(0) ==self.n_inputs
        self.mask = mask.detach().clone().unsqueeze(0) # (1, D)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = x * self.mask # broadcast (1, D)
        return self.net(x)


def init_D(opt, device):
    """Initialize discriminator
    """
    if opt.DiscArch == 'phiVpsi':
        D = D_phiVpsi([opt.Xdim,1], [opt.layerSizeX, opt.layerSizeY], opt.nonlin, opt.normalization).to(device)
    elif opt.DiscArch == 'concat':
        D = D_concat([opt.Xdim,1], opt.layerSizeX, opt.nonlin, opt.normalization).to(device) # no separate x,y branches
    elif opt.DiscArch == 'concat2':
        D = D_concat2([opt.Xdim, 1], opt.layerSize).to(device) # separate x,y branches then merge
    elif opt.DiscArch == 'concat_first':
        D = D_concat_first(sum([opt.Xdim, 1]), opt.layerSize, opt.dropout).to(device)
    elif opt.DiscArch == 'supervised':
        D = D_supervised(opt.Xdim, 1, opt.layerSize, opt.dropout).to(device)
    elif opt.DiscArch == 'supervised_nobias':
        D = D_supervised_nobias(opt.Xdim, 1, opt.layerSize, opt.dropout).to(device)
    return D


def init_optimizerD(opt, D, train_last_layer_only=False):
    """Initialize optimizer for discriminator D
    """
    params_to_train = D.parameters()
    if train_last_layer_only and opt.DiscArch == 'concat_first':
        params_to_train = D.last_linear.parameters()
    optimizerD = torch.optim.Adam(params_to_train, lr=opt.lrD, betas=(opt.beta1, opt.beta2), weight_decay=opt.wdecay)
    return optimizerD
