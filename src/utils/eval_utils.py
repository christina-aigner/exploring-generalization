import math

import numpy as np
from torch.utils.data import DataLoader

from measures.measures import *
from utils.model_utils import reparam


def validate(model, device, data_loader: DataLoader, criterion):
    """
    Calculates the loss and error of the model on a given data set.
    Args:
        model: model which should be evaluated
        device: cuda device
        data_loader: the DataLoader of the data set on which the model should be evaluated
        criterion: optimization criterion for the loss function

    Returns:

    """
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


def calc_exp_sharpness(model, init_model, device, train_loader, criterion):
    """
    Calculates the expected sharpness of a model, by drawing random perturbations from a
    Gaussian distribution.
    Args:
        model: model for which the sharpness should be analysed
        init_model: untrained base model
        device: cuda device
        train_loader: data loader of the training set
        criterion: optimizer of the loss function

    Returns:
        expected sharpness of the model.

    """
    clean_model = copy.deepcopy(model)
    clean_error, clean_loss, clean_margin = validate(clean_model, device, train_loader, criterion)
    calc_measure(model, init_model, add_gauss_perturbation, 'sharpness')
    pert_error, pert_loss, pert_margin = validate(model, device, train_loader, criterion)
    return pert_loss - clean_loss


def calc_measure(model, init_model, measure_func, operator, kwargs={}, p=1):
    """
    Provides a generic structure to calculate any measure given in 'measure_func' on the given model.
    This implementation is based on Neyshabur's repository on "generalization-bounds".
    Args:
        model: the model for which the measure should be calculated.
        init_model:  untrained base model.
        measure_func: the function of the measure, which should be computed
        operator: how the calculated function flows into the overall value of the measure.
        kwargs:
        p:

    Returns:
        the calculation of a given measure function for the given model.

    """
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


def calculate_norms(trained_model, init_model, device, margin, nchannels, img_dim):
    """
    Calculates l1_norm, l2_norm, spec_norm, l1_path and l2_path norm on a given model.

    Args:
        trained_model: the model for which the measure should be calculated.
        init_model: untrained base model
        device: cuda device
        trainingsetsize: trainingset size which the model was trained on
        margin: margin of the model
        nchannels: number of channels of the input image
        nclasses: number of classes for the softmax output
        img_dim: dimensions of the input image

    Returns:
        5-tuple of norms of the specified model
    """

    model = copy.deepcopy(trained_model)
    reparam(model)

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

    return l1_norm, l2_norm, spec_norm, l1_path, l2_path
