import argparse

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from data_utils import load_data
from models import vgg, fc
from models.model_utils import save_checkpoint

save_epochs = [1, 5, 10, 30, 50, 100, 200, 300, 400, 500, 600, 1000]

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
    parser.add_argument('--numhidden', default=1024, type=int,
                        help='number of hidden layers (default: 1024)')
    parser.add_argument('--trainingsetsize', default=50000, type=int,
                        help='size of the training set (options: 0 - 50k')

    # additional arguments
    parser.add_argument('--epochs', default=1000, type=int,
                        help='number of epochs to train (default: 600)')
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
    print("cuda available:", torch.cuda.is_available())
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    nchannels, nclasses, img_dim,  = 3, 10, 32

    # create an initial model
    if args.network == 'vgg':
        # customized vgg network
        model = vgg.Network(nchannels, nclasses)
    elif args.network == 'fc':
        # two layer perceptron
        model = fc.Network(nchannels, nclasses)
    else:
        raise ValueError("not a valid network argument.")

    model = model.to(device)

    # define loss function and optimizer
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.SGD(model.parameters(), learningrate, momentum=momentum)

    # loading data
    if args.randomlabels == True:
        train_dataset = load_data('train', 'CIFAR10RandomLabels', args.datadir)
        val_dataset = load_data('val', 'CIFAR10', args.datadir)
    else:
        train_dataset = load_data('train', 'CIFAR10', args.datadir)
        val_dataset = load_data('val', 'CIFAR10', args.datadir)

    print("trainings set size: ", args.trainingsetsize)

    train_subset = torch.utils.data.Subset(train_dataset, list(range(0, args.trainingsetsize)))
    # random seed with restricted size
    # sampler = torch.utils.data.SubsetRandomSampler(list(range(args.trainingsetsize)))
    # train_loader = DataLoader(train_dataset, batch_size=batchsize, sampler=sampler, **kwargs)
    train_loader = DataLoader(train_subset, batch_size=batchsize, shuffle=True, **kwargs)
    val_loader = DataLoader(val_dataset, batch_size=batchsize, shuffle=True, **kwargs)

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
                            f"../saved_models/checkpoint_{args.trainingsetsize}_{epoch}.pth")

        # stop training if the cross-entropy loss is less than the stopping condition
        if tr_loss < args.stopcond: break

    # calculate the training error and margin of the learned model
    tr_err, tr_loss, margin = validate(model, device, train_loader, criterion)
    save_checkpoint(epoch, model, optimizer, args.randomlabels, tr_loss, tr_err,
                    val_err, margin,
                    f"../saved_models/checkpoint_{args.trainingsetsize}_{epoch}.pth")

    print(f'\nFinal: Training loss: {tr_loss:.3f}\t Training margin {margin:.3f}\t ',
            f'Training error: {tr_err:.3f}\t Validation error: {val_err:.3f}\n')



if __name__ == '__main__':
    main()
