from copy import deepcopy
from typing import Dict

import torch
import torch.optim as optim
from torchvision import models

from models import vgg, fc


def save_model(model, PATH):
    """ Saves a model as state dict at a given Path"""
    torch.save(model.state_dict(), PATH)


def save_checkpoint(epoch, model, optimizer, random_labels, tr_loss, tr_error, val_error, margin,
                    PATH, targets):
    """
    Saves a training checkpoint with different pre-calculated parameters of the model. The main purpose of
    this approach is to strictly seperate the training from the model evaluation.

    Args:
        epoch: training epoch of the checkpoint
        model: trained model
        optimizer: used optimizer in the loss function
        random_labels: whether the model was trained on random labels
        tr_loss: calculated training loss
        tr_error: calculated training error
        val_error: calculated validation error
        margin: calculated margin
        PATH: path where the checkpoint should be stored
        targets: if random labels were used, this parameter stores the training targets, as random labels
            are set by random permutations, but need to be known during evaluation.

    """
    torch.save(
        {'epoch': epoch, 'model_state_dict': model.state_dict(), 'random_labels': random_labels,
         'optimizer_state_dict': optimizer.state_dict(), 'tr_loss': tr_loss, 'tr_error': tr_error,
         'val_error': val_error, 'margin': margin,
         'rand_targets': targets}, PATH)


def load_model(PATH, network='vgg', hiddenunits=1024, nchannels=3, nclasses=10):
    """
    Loads a model from a given path.
    Args:
        PATH: path where the model is located
        network: the type of the network
        hiddenunits: the number of hidden units in the network
        nchannels: number of channels of the dataset it has been trained on (1 for MNIST, 3 for CIFAR)
        nclasses: number of classes of the softmax output

    Returns:
        the loaded model in eval mode

    """
    if network == 'vgg':
        model = vgg.Network(nchannels, nclasses)
    elif network == 'fc':
        model = fc.Network(hiddenunits, nchannels, nclasses)
    else:
        raise ValueError("no valid network parameter.")

    if not torch.cuda.is_available():
        checkpoint = torch.load(PATH, map_location=lambda storage, loc: storage)
    else:
        checkpoint = torch.load(PATH)
    init_model = deepcopy(model)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model.eval(), init_model


def load_checkpoint_dict(PATH) -> Dict:
    """
    Loads a checkpoint from a given path.
    Args:
        PATH: path where checkpoint is located

    Returns:
        checkpoint dictionary
    """
    if not torch.cuda.is_available():
        checkpoint = torch.load(PATH, map_location=lambda storage, loc: storage)
    else:
        checkpoint = torch.load(PATH)
    return checkpoint


def load_checkpoint_train(PATH, network='vgg', hiddenunits=1024, nchannels=3,
                          nclasses=10, learningrate=0.01, momentum=0.9):
    """
    Loads a checkpoint for further training.
    Args:
        PATH: path where the model is located
        network: the type of the network
        hiddenunits: the number of hidden units in the network
        nchannels: number of channels of the dataset it has been trained on (1 for MNIST, 3 for CIFAR)
        nclasses: number of classes of the softmax output
        learningrate: leaning rate in the optimizer
        momentum: momentum rate in the optimizer

    Returns:
        the loaded model, optimizer and checkpoint dict

    """
    if not torch.cuda.is_available():
        checkpoint = torch.load(PATH, map_location=lambda storage, loc: storage)
    else:
        checkpoint = torch.load(PATH)

    if network == 'vgg':
        model = vgg.Network(nchannels, nclasses)
    elif network == 'fc':
        model = fc.Network(hiddenunits, nchannels, nclasses)
    else:
        raise ValueError("no valid network parameter.")

    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer = optim.SGD(model.parameters(), learningrate, momentum=momentum)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    model.train()

    return model, optimizer, checkpoint


def load_densenet():
    """
    Loads a pre-trained dense net.

    """
    return models.densenet161(pretrained=True)

def load_alexnet():
    """
    Loads a pre-trained alexnet.

    """
    return models.alexnet(pretrained=True)


def reparam(model, prev_layer=None):
    """
    Reparametrization of a network with batch normalization so that it calculates the same function as the
    original network but without batch normalization. Instead of removing batch norm completely, we set the bias and mean
    to zero, and scaling and variance to one.

    Args:
        model: input model to be reparameterized
        prev_layer: recursion helper, previous child of the model
    """
    for child in model.children():
        module_name = child._get_name()
        prev_layer = reparam(child, prev_layer)
        if module_name in ['Linear', 'Conv1d', 'Conv2d', 'Conv3d']:
            prev_layer = child
        elif module_name in ['BatchNorm2d', 'BatchNorm1d']:
            with torch.no_grad():
                scale = child.weight / ((child.running_var + child.eps).sqrt())
                prev_layer.bias.copy_( child.bias + (scale * (prev_layer.bias - child.running_mean)))
                perm = list(reversed(range(prev_layer.weight.dim())))
                prev_layer.weight.copy_((prev_layer.weight.permute(perm) * scale).permute(perm))
                child.bias.fill_(0)
                child.weight.fill_(1)
                child.running_mean.fill_(0)
                child.running_var.fill_(1)
    return prev_layer