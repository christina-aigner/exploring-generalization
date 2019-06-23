import argparse
import copy
from typing import List, Tuple

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from models import vgg, fc
from utils.data_utils import MNIST, CIFARSubset
from utils.eval_utils import validate
from utils.model_utils import save_checkpoint


def train(model, device, train_loader: DataLoader, loss_function, optimizer) -> Tuple:
    """
    Trains a model for one epoch.
    Args:
        args: python arguments
        model:
        device: cuda device
        train_loader:
        loss_function: loss function for optimization
        optimizer: optimizer for loss function

    Returns:
        training error and training loss in 2-Tuple

    """
    sum_loss, sum_correct = 0, 0

    # switch to train mode
    model.train()

    for i, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        # compute the output
        output = model(data)

        # compute the classification error and loss
        loss = loss_function(output, target)
        pred = output.max(1)[1]
        sum_correct += pred.eq(target).sum().item()
        sum_loss += len(data) * loss.item()

        # compute the gradient and do an SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        len_dataset = len(train_loader.dataset)

    return 1 - (sum_correct / len_dataset), (sum_loss / len_dataset)


def main():

    # settings
    parser = argparse.ArgumentParser(description='Training a VGG Net')
    # arguments needed for experiments
    parser.add_argument('--network', default='vgg', type=str,
                        help='type of network (options: vgg | fc, default: vgg)')
    parser.add_argument('--numhidden', default=None, type=int,
                        help='number of hidden layers (default: 1024)')
    parser.add_argument('--trainingsetsize', default=60000, type=int,
                        help='size of the training set if CIFAR10 is used (options: 0 - 50k')
    parser.add_argument('--dataset', default=MNIST,
                        help='load data set for evaluation if sharpness should be calculated, for norms, '
                             'this is not necessary (options: MNIST | CIFARSubset')
    # additional arguments
    parser.add_argument('--randomlabels', default=False, type=bool,
                        help='training with random labels Yes or No? (options: True | False, default: False)')
    parser.add_argument('--saveepochs', default=[], type=List,
                        help='epochs which should be saved as checkpoint, last epoch is always saved.')
    parser.add_argument('--epochs', default=1000, type=int,
                        help='number of epochs to train (default: 1000)')
    parser.add_argument('--stopcond', default=0.01, type=float,
                        help='stopping condition based on the training error (default: 0.01)')
    parser.add_argument('--datadir', default='../datasets', type=str,
                        help='path to the directory that contains the datasets (default: datasets)')

    args = parser.parse_args()

    # fixed parameters
    save_epochs = args.saveepochs
    learningrate = 0.01
    momentum = 0.9

    # cuda settings
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print("cuda available:", use_cuda)
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    # load data
    if args.dataset == 'MNIST':
        train_loader, val_loader = MNIST(args, **kwargs)
        nchannels = 1
    elif args.dataset == 'CIFARSubset':
        train_loader, val_loader = CIFARSubset(args, **kwargs)
        nchannels = 3
    else:
        raise ValueError('a dataset has to be specified in the arguments.')

    nclasses, img_dim, = 10, 32

    # create an initial model
    if args.network == 'vgg':
        # customized vgg network
        init_model = vgg.Network(nchannels, nclasses)
    elif args.network == 'fc':
        # two layer perceptron
        init_model = fc.Network(args.numhidden, nchannels, nclasses)
    else:
        raise ValueError("not a valid network argument.")

    model = copy.deepcopy(init_model).to(device)

    # define loss function and optimizer
    loss_function = nn.CrossEntropyLoss().to(device)
    optimizer = optim.SGD(model.parameters(), learningrate, momentum=momentum)

    # if we train random labels, we add the random label tensor in the checkpoint
    used_targets = None
    if args.randomlabels == True:
        used_targets = train_loader.dataset.dataset.targets

    # training the model
    for epoch in range(0, args.epochs):
        # train for one epoch
        tr_err, tr_loss = train(model, device, train_loader, loss_function, optimizer)

        val_err, val_loss, val_margin = validate(model, device, val_loader, loss_function)

        print(f'Epoch: {epoch + 1}/{args.epochs}\t Training loss: {tr_loss:.3f}\t',
                f'Training error: {tr_err:.3f}\t Validation error: {val_err:.3f}')

        if epoch in save_epochs:
            save_checkpoint(epoch, model, optimizer, args.randomlabels, tr_loss, tr_err,
                            val_err, val_margin,
                            f"../saved_models/cp_{args.network}_{args.numhidden}_{args.trainingsetsize}_{epoch}.pth",
                            used_targets)

        # stop training if the cross-entropy loss is less than the stopping condition
        if tr_err < args.stopcond: break

    # validate the model and calculate the margin
    tr_err, tr_loss, margin = validate(model, device, train_loader, loss_function)

    # save the trained model with all important parameters for later evaluation or retraining
    save_checkpoint(epoch, model, optimizer, args.randomlabels, tr_loss, tr_err,
                    val_err, margin,
                    f"../saved_models/cp_{args.network}_{args.numhidden}_{args.trainingsetsize}_{epoch}.pth",
                    used_targets)

    print(f'\nFinal: Training loss: {tr_loss:.3f}\t Training margin {margin:.3f}\t ',
            f'Training error: {tr_err:.3f}\t Validation error: {val_err:.3f}\n')


if __name__ == '__main__':
    main()
