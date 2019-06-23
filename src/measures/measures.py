import copy

import torch
from torch.distributions import normal


def add_gauss_perturbation(module, alpha=5e-4):
    std = alpha * (10 * torch.abs(torch.mean(module.weight.data)) + 1)
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


# This function calculates path-norm introduced in Neyshabur et al. 2015
def lp_path_norm(model, device, p=2, input_size=[3, 32, 32]):
    tmp_model = copy.deepcopy(model)
    tmp_model.eval()
    for param in tmp_model.parameters():
        if param.requires_grad:
            param.abs_().pow_(p)
    data_ones = torch.ones(input_size).to(device)
    return (tmp_model(data_ones).sum() ** (1 / p)).item()




