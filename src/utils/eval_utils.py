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


def calc_exp_sharpness(model, device, train_loader, criterion):
    """
    Calculates the expected sharpness of a model, by drawing random perturbations from a
    Gaussian distribution.
    Args:
        model: model for which the sharpness should be analysed
        device: cuda device
        train_loader: data loader of the training set
        criterion: optimizer of the loss function

    Returns:
        expected sharpness of the model.

    """
    clean_model = copy.deepcopy(model)
    clean_error, clean_loss, clean_margin = validate(clean_model, device, train_loader, criterion)
    model_sharpness(model)
    pert_error, pert_loss, pert_margin = validate(model, device, train_loader, criterion)
    return pert_loss - clean_loss


def calculate_norms(trained_model, device, margin, nchannels, img_dim):
    """
    Calculates l1_norm, l2_norm, spec_norm, l1_path and l2_path norm on a given model.

    Args:
        trained_model: the model for which the measure should be calculated.
        device: cuda device
        trainingsetsize: training set size which the model was trained on
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
        l2_norm = math.exp(l_norm(model, p=2, q=2)) / margin
        spec_norm = math.exp(spectral(model)) / margin

        l1_path = path_norm(model, device, p=1,
                            input_size=[1, nchannels, img_dim, img_dim]) / margin
        l2_path = path_norm(model, device, p=2,
                            input_size=[1, nchannels, img_dim, img_dim]) / margin

    return l2_norm, spec_norm, l1_path, l2_path
