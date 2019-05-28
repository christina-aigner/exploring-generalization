import math

import numpy as np
from torch.utils.data import DataLoader

from measures.measures_old import *


# This function reparametrizes the networks with batch normalization in a way that it calculates the same function as the
# original network but without batch normalization. Instead of removing batch norm completely, we set the bias and mean
# to zero, and scaling and variance to one
# Warning: This function only works for convolutional and fully connected networks. It also assumes that
# module.children() returns the children of a module in the forward pass order. Recurssive construction is allowed.


# evaluate the model on the given set
def validate(model, device, data_loader: DataLoader, criterion):
    sum_loss, sum_correct = 0, 0
    margin = torch.Tensor([]).to(device)

    # switch to evaluation mode
    model.eval()
    with torch.no_grad():
        for i, (data, target) in enumerate(data_loader):
            data, target = data.to(device), target.to(device)

            # compute the output
            output = model(data)

            # compute the classification error and loss
            pred = output.max(1)[1]
            sum_correct += pred.eq(target).sum().item()
            sum_loss += len(data) * criterion(output, target).item()

            # compute the margin
            output_m = output.clone()
            for i in range(target.size(0)):
                output_m[i, target[i]] = output_m[i, :].min()
            margin = torch.cat((margin, output[:, target].diag() - output_m[:,
                                                                   output_m.max(1)[
                                                                       1]].diag()),
                               0)
        margin = np.percentile(margin.cpu().numpy(), 5)

        len_dataset = len(data_loader.dataset)

    return 1 - (sum_correct / len_dataset), (sum_loss / len_dataset), margin


def reparam(model, prev_layer=None):
    for child in model.children():
        module_name = child._get_name()
        prev_layer = reparam(child, prev_layer)
        if module_name in ['Linear', 'Conv1d', 'Conv2d', 'Conv3d']:
            prev_layer = child
        elif module_name in ['BatchNorm2d', 'BatchNorm1d']:
            with torch.no_grad():
                scale = child.weight / ((child.running_var + child.eps).sqrt())
                prev_layer.bias.copy_(
                    child.bias + (scale * (prev_layer.bias - child.running_mean)))
                perm = list(reversed(range(prev_layer.weight.dim())))
                prev_layer.weight.copy_(
                    (prev_layer.weight.permute(perm) * scale).permute(perm))
                child.bias.fill_(0)
                child.weight.fill_(1)
                child.running_mean.fill_(0)
                child.running_var.fill_(1)
    return prev_layer


def calc_sharpness(model, init_model, device, train_loader, criterion):
    clean_model = copy.deepcopy(model)
    clean_error, clean_loss, clean_margin = validate(clean_model, device, train_loader, criterion)
    calc_measure(model, init_model, add_gauss_perturbation, 'sharpness')
    pert_error, pert_loss, pert_margin = validate(model, device, train_loader, criterion)
    return pert_loss - clean_loss


# This function calculates a measure on the given model
# measure_func is a function that returns a value for a given linear or convolutional layer
# calc_measure calculates the values on individual layers and then calculate the final value based on the given operator
def calc_measure(model, init_model, measure_func, operator, kwargs={}, p=1):
    measure_val = 0
    if operator == 'product':
        measure_val = math.exp(
            calc_measure(model, init_model, measure_func, 'log_product', kwargs, p))
    elif operator == 'norm':
        measure_val = (calc_measure(model, init_model, measure_func, 'sum', kwargs,
                                    p=p)) ** (1 / p)
    else:
        measure_val = 0
        for child, init_child in zip(model.children(), init_model.children()):
            module_name = child._get_name()
            if module_name in ['Linear', 'Conv1d', 'Conv2d', 'Conv3d']:
                if operator == 'log_product':
                    measure_val += math.log(measure_func(child, init_child, **kwargs))
                elif operator == 'sum':
                    measure_val += (measure_func(child, init_child, **kwargs)) ** p
                elif operator == 'max':
                    measure_val = max(measure_val,
                                      measure_func(child, init_child, **kwargs))
                elif operator == 'sharpness':
                    measure_func(child)
            else:
                measure_val += calc_measure(child, init_child, measure_func, operator,
                                            kwargs, p=p)
    return measure_val


# This function calculates various measures on the given model and returns two dictionaries:
# 1) measures: different norm based measures on the model
# 2) bounds: different generalization bounds on the model
def calculate(trained_model, init_model, device, trainingsetsize, margin, nchannels,
              nclasses, img_dim):
    model = copy.deepcopy(trained_model)
    reparam(model)
    reparam(init_model)

    # size of the training set
    m = trainingsetsize

    # depth
    d = calc_measure(model, init_model, depth, 'sum', {})

    # number of parameters (not including batch norm)
    nparam = calc_measure(model, init_model, n_param, 'sum', {})

    measure, bound = {}, {}
    with torch.no_grad():
        l1_norm = calc_measure(model, init_model, norm, 'product',
                               {'p': 1, 'q': float('Inf')}) / margin
        l2_norm = calc_measure(model, init_model, norm, 'product',
                               {'p': 2, 'q': 2}) / margin
        spec_norm = calc_measure(model, init_model, op_norm, 'product',
                                 {'p': float('Inf')}) / margin
        l1_path = lp_path_norm(model, device, p=1,
                               input_size=[1, nchannels, img_dim, img_dim]) / margin
        l2_path = lp_path_norm(model, device, p=2,
                               input_size=[1, nchannels, img_dim, img_dim]) / margin

        measure['L_{3,1.5} norm'] = calc_measure(model, init_model, norm, 'product',
                                                 {'p': 3, 'q': 1.5}) / margin
        measure['L_1.5 operator norm'] = calc_measure(model, init_model, op_norm,
                                                      'product', {'p': 1.5}) / margin
        measure['Trace norm'] = calc_measure(model, init_model, op_norm, 'product',
                                             {'p': 1}) / margin
        measure['L1.5_path norm'] = lp_path_norm(model, device, p=1.5,
                                                 input_size=[1, nchannels, img_dim,
                                                             img_dim]) / margin

        # Generalization bounds: constants and additive logarithmic factors are not included
        # This value of alpha is based on the improved depth dependency by Golowith et al. 2018
        alpha = math.sqrt(d + math.log(nchannels * img_dim * img_dim))
        # L1_max Bound (Bartlett and Mendelson 2002)
        l1_max_bound = (alpha * l1_norm / math.sqrt(m))
        # Frobenious Bound (Neyshabur et al. 2015)
        frobenius_bound = (alpha * l2_norm / math.sqrt(m))

        # 'Spec_Fro Bound (Neyshabur et al. 2018)'
        ratio = calc_measure(model, init_model, h_dist_op_norm, 'norm',
                             {'p': 2, 'q': 2, 'p_op': float('Inf')}, p=2)
        spec_l2_bound = (d * spec_norm * ratio / math.sqrt(m))

    return l1_norm, l2_norm, spec_norm, l1_path, l2_path, l1_max_bound, frobenius_bound, spec_l2_bound
