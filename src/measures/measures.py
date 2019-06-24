import copy
import math

import torch
from torch.distributions import normal


def model_sharpness(model):
    """
    Calculates sharpness of a model by adding a perturbation to each module.
    The "add_gauss_perturbation" function can be substituted with another perturbation generator.
    Args:
        model: model to be evaluated

    Returns:
        sharpness value
    """
    result = 0
    for child in model.children():
        module_name = child._get_name()
        if module_name in ['Linear', 'Conv1d', 'Conv2d', 'Conv3d']:
            result += add_gauss_perturbation(child)
        else:
            result += model_sharpness(child)


def l_norm(model, p=2, q=2.0):
    """
    Calculates a l-norm for a model.
    Args:
        model: model for which the norm should be calculated
        p: p-value
        q: p-value

    Returns:

    """
    result = 0
    for child in model.children():
        module_name = child._get_name()
        if module_name in ['Linear', 'Conv1d', 'Conv2d', 'Conv3d']:
            result += math.log(norm(child, p, q))
        else:
            result += l_norm(child, p, q)

    return result


def spectral(model, p=float('Inf')):
    """
    Calculates the spectral norm for a model.
    Args:
        model: model for which the norm should be calculated.
        p: p-value

    Returns:

    """
    result = 0
    for child in model.children():
        module_name = child._get_name()
        if module_name in ['Linear', 'Conv1d', 'Conv2d', 'Conv3d']:
            result += math.log(spectral_norm(child, p))
        else:
            result += spectral(child, p)

    return result


def path_norm(model, device, p=2, input_size=[3, 32, 32]):
    """
    calculates the path norm of a weight matrix of a module.
    Args:
        model: module for which the norm should be calculated
        device: cuda device
        p: p value for norm
        input_size: input dimension of the model

    Returns:
        norm value
    """
    tmp_model = copy.deepcopy(model)
    tmp_model.eval()
    for param in tmp_model.parameters():
        if param.requires_grad:
            param.abs_().pow_(p)
    data_ones = torch.ones(input_size).to(device)
    return (tmp_model(data_ones).sum() ** (1 / p)).item()


def add_gauss_perturbation(module, alpha=5e-4):
    """
    Adds a randomly drawn gaussian perturbation to the weight matrix of a module.
    Args:
        module: module for which the perturbation should be added
        alpha: perturbation bound

    """
    std = alpha * (10 * torch.abs(torch.mean(module.weight.data)) + 1)
    m = normal.Normal(0, std)
    perturbation = m.sample((1))
    module.weight.data = module.weight.data + perturbation


def norm(module, p=2, q=2):
    """
    Calculates the l-norm of the weight matrix of a module
    Args:
        module: module for which the norm should be calculated
        p: p value for norm
        q: q value for norm

    Returns:
        norm value

    """
    reshaped = module.weight.view(module.weight.size(0), -1)
    return reshaped.norm(p=p, dim=1).norm(q).item()


def spectral_norm(module, p=float('Inf')):
    """
    Calculates the norm of eigen values of a module
    Args:
        module: module for which the norm should be calculated
        p: p value for norm

    Returns:
        norm value
    """
    reshaped = module.weight.view(module.weight.size(0), -1)
    _, S, _ = reshaped.svd()
    return S.norm(p).item()





