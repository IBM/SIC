import torch
import torch.nn.functional as F
from utils import log_current_variables


def minibatch(data, batch_size, requires_grad=True):
    x, y = data
    if not batch_size:
        x, y = x.clone(), y.clone()
        return x.requires_grad_(requires_grad), y.requires_grad_(requires_grad)
    else:
        indx = torch.LongTensor(batch_size).random_(0, x.size(0))
        return x[indx].requires_grad_(requires_grad), y[indx].requires_grad_(requires_grad)


def compute_objective_terms(data, net, need_penalty_terms=False):
    """Construct minibatch of (x,y): compute main objective term and optionally penalty terms.
        Returns expectations over minibatch (last 2 terms only if penalty_terms=True)
        * E[ f(x,y) ]
        * E[ f(x,y)**2 ]
        * [ E[ |df/dx_j|^2 ], E[ |df/dy_j|^2 ] ]
    """
    device = next(net.parameters()).device
    x = data[0].requires_grad_(need_penalty_terms).to(device)
    y = data[1].requires_grad_(False).to(device)
    f = net(x,y)
    E_f = f.mean(0)

    E_f2, E_grad2 = None, None
    if need_penalty_terms:
        E_f2 = (f**2).sum()
        gradx = torch.autograd.grad(f.sum(), x, create_graph=True)[0]
        E_grad2 = (gradx**2).mean(0) # expectation, keep x_j coordinates separate.
    return E_f, E_f2, E_grad2


def compute_mse(dataloader, net, targets_mu=0.0, targets_sd=1.0):
    device = next(net.parameters()).device
    net.eval()
    mse_loss = 0
    n_samples = 0
    with torch.no_grad():
        for data, targets in dataloader:
            data, targets = data.to(device), targets.to(device)
            targets_norm = (targets.view(-1) - targets_mu) / targets_sd
            mse_loss += F.mse_loss(net(data).view(-1), targets_norm, reduction='sum')
            n_samples += data.shape[0]
    return mse_loss / n_samples


def sobolev_forward(D, eta_x, dataP, dataQ, mu, selectSobo='L1^2'):
    ETA_EPS = 1e-6  # stabilize denominator in eta constraints

    Ep_f, Ep_f2, Ep_grad2 = compute_objective_terms(dataP, D, need_penalty_terms='P' in mu)
    Eq_f, Eq_f2, Eq_grad2 = compute_objective_terms(dataQ, D, need_penalty_terms='Q' in mu)

    # mu: dominant measure on which to compute expectations.
    if mu == 'P':
        Emu_f2, Emu_grad2 = Ep_f2, Ep_grad2
    elif mu == 'Q':
        Emu_f2, Emu_grad2 = Eq_f2, Eq_grad2
    elif mu == 'P+Q/2':
        Emu_f2, Emu_grad2 = (Ep_f2 + Eq_f2) / 2, (Ep_grad2 + Eq_grad2) / 2

    sobo_dist = (Ep_f - Eq_f)
    constraint_f2 = Emu_f2

    if selectSobo == 'L2':
        constraint_Sobo = Emu_grad2.sum()  # L2 norm: sum_j (E |df / dx_j |^2)
    elif selectSobo == 'L1-biased':
        constraint_Sobo = Emu_grad2.sqrt().sum()  # L1 norm: sum_j sqrt(E |df / dx_j |^2)
    elif selectSobo == 'L1':
        constraint_Sobo = (Emu_grad2 / (eta_x + ETA_EPS)).sum() + eta_x.sum()
    elif selectSobo == 'L1^2':
        constraint_Sobo = (Emu_grad2 / (eta_x + ETA_EPS)).sum()
    else:
        raise KeyError('Unrecognized selectSobo argument == {}'.format(selectSobo))

    return sobo_dist, constraint_f2, constraint_Sobo


def Ep_D(D, test_loader):
    device = next(D.parameters()).device
    D.eval()
    Ep_f = []
    with torch.no_grad():
        for batch_idx, (data, targets) in enumerate(test_loader):
            data, targets = data.to(device), targets.to(device)
            Ep_f.append(D(data, targets))
    return torch.cat(Ep_f).mean(0).item()


def avg_sobolev_dist(D, dl_P, dl_Q):
    D.eval()
    sobo_dist = []
    n_samples = 0
    with torch.no_grad():
        for (dataP, dataQ) in zip(dl_P, dl_Q):

            Ep_f, Ep_f2, Ep_grad2 = compute_objective_terms(dataP, D)
            Eq_f, Eq_f2, Eq_grad2 = compute_objective_terms(dataQ, D)

            sobo_dist.append((Ep_f - Eq_f) * dataP[0].shape[0])
            n_samples += dataP[0].shape[0]

    return torch.cat(sobo_dist).sum(0).item() / n_samples


def compute_objective_supervised(inputs, targets, net):
    """Construct minibatch of (x,y): compute main objective term and optionally penalty terms.
        Returns expectations over minibatch (last 2 terms only if penalty_terms=True)
        * E[ |df/dx_j|^2 ]
    """
    device = next(net.parameters()).device
    inputs, targets = inputs.requires_grad_(True).to(device), targets.requires_grad_(False).to(device)

    outputs = net(inputs).view(-1)
    mse_loss = F.mse_loss(outputs, targets)

    gradx = torch.autograd.grad(outputs.sum(), inputs, create_graph=True)[0]
    E_grad2 = (gradx**2).mean(0)  # expectation, keep x_j coordinates separate.
    return mse_loss, E_grad2


def supervised_forward_sobolev_penalty(net, inputs, targets, eta_x, selectSobo='L1^2'):
    ETA_EPS = 1e-6  # stabilize denominator in eta constraints

    mse_loss, Ep_grad2 = compute_objective_supervised(inputs, targets, net)

    if selectSobo == 'L2':
        constraint_Sobo = Ep_grad2.sum()  # L2 norm: sum_j (E |df / dx_j |^2)
    elif selectSobo == 'L1-biased':
        constraint_Sobo = Ep_grad2.sqrt().sum()  # L1 norm: sum_j sqrt(E |df / dx_j |^2)
    elif selectSobo == 'L1':
        constraint_Sobo = (Ep_grad2 / (eta_x + ETA_EPS)).sum() + eta_x.sum()
    elif selectSobo == 'L1^2':
        constraint_Sobo = (Ep_grad2 / (eta_x + ETA_EPS)).sum()
    else:
        raise KeyError('Unrecognized selectSobo argument == {}'.format(selectSobo))

    return mse_loss, constraint_Sobo


def recompute_etas_P(D, dataloader):
    """Recomputes etas integrating dD/dx over P
    """
    etas = 0
    n_samples = 0
    for data in dataloader:
        _, _, Ep_grad2 = compute_objective_terms(data, D, need_penalty_terms=True)
        # Only 1 term, i.e. integrate over P ((P+Q)/2 would work also)
        etas += Ep_grad2.detach() * data[0].shape[0]
        n_samples += data[0].shape[0]
    return etas / n_samples


def normalize_etas(eta):
    EPS = 1e-6
    logits = (eta.data + EPS).log()
    eta.data.copy_(torch.softmax(logits, 0))
    return eta


def log_eta_stats(tbw, t, eta_x, eta_lr, tb_trunc_tensor_size):
    ETA_EPS = 1e-6  # stabilize denominator in eta constraints

    eta_x_grad = eta_x.grad
    # Linf norm: max eta update, how far from 1 on average
    eta_x_update = (torch.exp(-eta_lr * eta_x.grad) - 1).abs()
    eta_x_update_L1 = eta_x_update.sum()
    eta_x_update_Linf = eta_x_update.max()
    if torch.isclose(eta_x.data.sum(), torch.tensor(1.0)):
        # entropy if etas are on the simplex
        eta_entropy = - (eta_x.data * eta_x.data.log()).sum()
    eta_sparse_count = (eta_x < 10 * ETA_EPS).sum()
    log_current_variables(tbw, t, locals(),
            keys_to_log=['eta_x', 'eta_x_grad', 'eta_x_update_L1', 'eta_x_update_Linf',
                'eta_entropy', 'eta_sparse_count'],
            key_prefix='eta/',
            tb_trunc_tensor_size=tb_trunc_tensor_size)


def logstab_mirror_descent_step_(eta, lr):
    # V3: stabilized in log domain, based on Youssef chat in channel on May 1st.
    EPS = 1e-6
    logits = (eta.data + EPS).log()
    logits.add_(-lr, eta.grad.data)
    eta.data.copy_(torch.softmax(logits, 0))


def reduced_gradient_step_(eta, lr):
    """Reduced gradient for projecting on the simplex
        See Bonnans, used in SimpleMKL paper: http://www.jmlr.org/papers/volume9/rakotomamonjy08a/rakotomamonjy08a.pdf
        Note: initialize eta to uniform for this to work
    """
    # get the maximum value of eta
    eta_max,index_max = torch.max(eta.data,0)

    # define reduced gradient
    reduced_grad = eta.grad.data
    grad_eta_max = eta.grad.data[index_max]
    reduced_grad = reduced_grad - grad_eta_max.expand_as(reduced_grad)
    reduced_grad[index_max] = - torch.sum(reduced_grad)
    ## find if any eta = 0 and its reduced gardient positive
    intersection = (eta == 0)*(reduced_grad > 0)
    indices_intersection =  intersection.nonzero()
    ## reduced gradient update
    reduced_grad_corrected = reduced_grad
    reduced_grad_corrected[indices_intersection] = 0
    reduced_grad_corrected[index_max] = 0
    reduced_grad_corrected[index_max] = - torch.sum(reduced_grad_corrected)

    ### apply gradient descent with reduced gradient descent
    x = -lr * reduced_grad_corrected
    eta.data.add_(x)
    # End of training - evaluate etas


def eta_optim_step_(eta, eta_step_type, lr):
    """Optimization step for etas
        Note: this only implements 'L1^2'
    """
    if eta_step_type == 'mirror':
        logstab_mirror_descent_step_(eta, lr)
    elif eta_step_type == 'reduced':
        reduced_gradient_step_(eta, lr)
    else:
        raise KeyError("eta_step_type must be one of the following values: mirror | reduced" )
    return eta


def heldout_eval(tbw, t, te_P, te_Q, D, logger=None):
    Ep_f, Ep_f2, Ep_grad2 = compute_objective_terms(te_P, D, need_penalty_terms=True)
    Eq_f, Eq_f2, Eq_grad2 = compute_objective_terms(te_Q, D, need_penalty_terms=True)
    sobo_dist = Ep_f.item() - Eq_f.item()
    # NOTE  were taking mu=Q here, should be passed in from opt.mu
    betas = Eq_grad2
    constraint_f2 = Eq_f2
    constraint_L2 = Eq_grad2.sum() # L2 norm: sum_j (E |df / dx_j |^2)
    constraint_L1 = Eq_grad2.sqrt().sum() # L1 norm: sum_j sqrt(E |df / dx_j |^2)
    log_current_variables(tbw, t, locals(),
            keys_to_log=['sobo_dist', 'betas', 'constraint_f2', 'constraint_L2', 'constraint_L1'],
            key_prefix='hld/')

    msg = '[{:5d}] sobo_dist={:.4f} constraint_L2={:.4f} constraint_L1={:.4f} constraint_f2={:.4f} Ep_f={:.4f} Eq_f={:.4f}'.format(
        t, sobo_dist, constraint_L2.item(), constraint_L1.item(), constraint_f2.item(), Ep_f.item(), Eq_f.item())

    if logger: logger.info(msg)
