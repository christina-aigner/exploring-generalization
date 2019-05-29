import copy

import torch
from torch.distributions import normal


# This function reparametrizes the networks with batch normalization in a way that it calculates the same function as the
# original network but without batch normalization. Instead of removing batch norm completely, we set the bias and mean
# to zero, and scaling and variance to one
# Warning: This function only works for convolutional and fully connected networks. It also assumes that
# module.children() returns the children of a module in the forward pass order. Recurssive construction is allowed.

class Sharpness:
    def __init__(self):
        self.A = None
        self.upper = None
        self.lower = None
        self.current_pert = None
        self.clean_error = None
        self.sharpness = None

    def update_bounds(self, module, alpha=5e-4):
        upper = alpha * (module.weight.data + 1)
        lower = -alpha * (module.weight.data + 1)
        if upper > self.upper:
            self.upper = upper
        if lower < self.lower:
            self.lower = lower

    def add_perturbation(self, module, v):
        module.weight.data = self.A * v


def add_random_perturbation(module, alpha=5e-4):
    """
    add perturbation v = alpha * (|w| + 1) to a module
    Args:
        module:

    Returns:

    """
    upper_bound = alpha * (torch.abs(module.weight.data) + 1)
    module.weight.data = torch.tensor.random_(0, upper_bound)


def add_gauss_perturbation(module, alpha=5e-4):
    std = alpha * (10 * torch.abs(module.weight.data) + 1)
    m = normal.Normal(0, std)
    perturbation = m.sample((1))
    module.weight.data = module.weight.data + perturbation


# calculates l_pq norm of the parameter matrix of a layer:
# 1) l_p norm of incomming weights to each hidden unit and l_q norm on the hidden units
# 2) convolutional tensors are reshaped in a way that all dimensions except the output are together
def norm(module, init_module, p=2, q=2):
    return module.weight.view(module.weight.size(0), -1).norm(p=p, dim=1).norm(q).item()


# calculates l_p norm of eigen values of a layer
# convolutional tensors are reshaped in a way that all dimensions except the output are together
def op_norm(module, init_module, p=float('Inf')):
    _, S, _ = module.weight.view(module.weight.size(0), -1).svd()
    return S.norm(p).item()


# calculates l_pq distance of the parameter matrix of a layer from the random initialization:
# 1) l_p norm of incomming weights to each hidden unit and l_q norm on the hidden units
# 2) convolutional tensors are reshaped in a way that all dimensions except the output are together
def dist(module, init_module, p=2, q=2):
    return (module.weight - init_module.weight).view(module.weight.size(0), -1).norm(
        p=p, dim=1).norm(q).item()


# calculates l_pq distance of the parameter matrix of a layer from the random initialization with an extra factor that
# depends on the number of hidden units
def h_dist(module, init_module, p=2, q=2):
    return (n_hidden(module, init_module) ** (1 - 1 / q)) * dist(module, init_module,
                                                                 p=p, q=q)


# ratio of the h_dist to the operator norm
def h_dist_op_norm(module, init_module, p=2, q=2, p_op=float('Inf')):
    return h_dist(module, init_module, p=p, q=q) / op_norm(module, init_module, p=p_op)


# number of hidden units
def n_hidden(module, init_module):
    return module.weight.size(0)


# depth --> always 1 for any linear of convolutional layer
def depth(module, init_module):
    return 1

# number of parameters
def n_param(module, init_module):
    bparam = 0 if module.bias is None else module.bias.size(0)
    return bparam + module.weight.size(0) * module.weight.view(module.weight.size(0),
                                                               -1).size(1)

# This function calculates path-norm introduced in Neyshabur et al. 2015
def lp_path_norm(model, device, p=2, input_size=[3, 32, 32]):
    tmp_model = copy.deepcopy(model)
    tmp_model.eval()
    for param in tmp_model.parameters():
        if param.requires_grad:
            param.abs_().pow_(p)
    data_ones = torch.ones(input_size).to(device)
    return (tmp_model(data_ones).sum() ** (1 / p)).item()




