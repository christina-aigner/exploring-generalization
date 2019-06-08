import argparse
import copy

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from models import vgg, fc
from utils.data_utils import MNIST
from utils.eval_utils import validate
from utils.model_utils import save_checkpoint

save_epochs = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500]


def train(model, device, train_loader: DataLoader, criterion, optimizer):
    """
    Train a model for one epoch
    Args:
        args:
        model:
        device:
        train_loader:
        criterion:
        optimizer:
        random_labels:

    Returns:

    """
    sum_loss, sum_correct = 0, 0

    # switch to train mode
    model.train()

    for i, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        # compute the output
        output = model(data)

        # compute the classification error and loss
        loss = criterion(output, target)
        pred = output.max(1)[1]
        sum_correct += pred.eq(target).sum().item()
        sum_loss += len(data) * loss.item()

        # compute the gradient and do an SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        len_dataset = len(train_loader.dataset)

    return 1 - (sum_correct / len_dataset), (sum_loss / len_dataset)


# This function trains a neural net on the given dataset and calculates various measures on the learned network.
def main():

    # settings
    parser = argparse.ArgumentParser(description='Training a VGG Net')
    # arguments needed for experiments
    parser.add_argument('--network', default='vgg', type=str,
                        help='type of network (options: vgg | fc, default: vgg)')
    parser.add_argument('--randomlabels', default=False, type=bool,
                        help='training with random labels Yes or No? (options: True | False, default: False)')
    parser.add_argument('--numhidden', default=None, type=int,
                        help='number of hidden layers (default: 1024)')
    parser.add_argument('--trainingsetsize', default=60000, type=int,
                        help='size of the training set (options: 0 - 50k')

    # additional arguments
    parser.add_argument('--epochs', default=1000, type=int,
                        help='number of epochs to train (default: 800)')
    parser.add_argument('--stopcond', default=0.01, type=float,
                        help='stopping condtion based on the cross-entropy loss (default: 0.01)')
    parser.add_argument('--no-cuda', default=False, action='store_true',
                        help='disables CUDA training')
    parser.add_argument('--datadir', default='../datasets', type=str,
                        help='path to the directory that contains the datasets (default: datasets)')

    args = parser.parse_args()

    # fixed parameters
    batchsize = 64
    learningrate = 0.01
    momentum = 0.9

    # cuda settings
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print("cuda available:", )
    use_cuda = torch.cuda.is_available()
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    nchannels, nclasses, img_dim, = 1, 10, 32

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
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.SGD(model.parameters(), learningrate, momentum=momentum)

    # load data
    train_loader, val_loader = MNIST(args, **kwargs)
    used_targets = None
    if args.randomlabels == True:
        used_targets = train_loader.dataset.targets

    # training the model
    for epoch in range(0, args.epochs):
        # train for one epoch
        tr_err, tr_loss = train(model, device, train_loader, criterion, optimizer)

        val_err, val_loss, val_margin = validate(model, device, val_loader, criterion)

        print(f'Epoch: {epoch + 1}/{args.epochs}\t Training loss: {tr_loss:.3f}\t',
                f'Training error: {tr_err:.3f}\t Validation error: {val_err:.3f}')

        if epoch in save_epochs:
            save_checkpoint(epoch, model, optimizer, args.randomlabels, tr_loss, tr_err,
                            val_err, val_margin,
                            f"../saved_models/cp_{args.network}_{args.numhidden}_{args.trainingsetsize}_{epoch}.pth",
                            used_targets)

        # stop training if the cross-entropy loss is less than the stopping condition
        if tr_err < args.stopcond: break

    # calculate the training error and margin of the learned model
    tr_err, tr_loss, margin = validate(model, device, train_loader, criterion)
    # sharpness = calc_sharpness(model, init_model, device, train_loader, criterion)

    save_checkpoint(epoch, model, optimizer, args.randomlabels, tr_loss, tr_err,
                    val_err, margin,
                    f"../saved_models/cp_{args.network}_{args.numhidden}_{args.trainingsetsize}_{epoch}.pth",
                    used_targets)

    print(f'\nFinal: Training loss: {tr_loss:.3f}\t Training margin {margin:.3f}\t ',
            f'Training error: {tr_err:.3f}\t Validation error: {val_err:.3f}\n')


if __name__ == '__main__':
    main()
