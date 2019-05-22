import copy
import torch
from torch import nn, optim
from models.vgg import Network
from torch.utils.data import RandomSampler, DataLoader
from data_utils import load_data
def main():

    trainingsetsize = 50000
    random_labels = False
    dataset = 'CIFAR10'
    datadir = '../datasets'
    # fixed parameters
    batchsize = 64
    learningrate = 0.01
    momentum = 0.9

    nchannels, nclasses, img_dim,  = 3, 10, 32

    # create an initial model
    #model = getattr(importlib.import_module('.models.{}'.format(args.model)), 'Network')(nchannels, nclasses)

    model = Network(nchannels, nclasses)


    # create a copy of the initial model to be used later
    init_model = copy.deepcopy(model)

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), learningrate, momentum=momentum)

    # loading data
    train_dataset = load_data('train', dataset, datadir, nchannels)
    val_dataset = load_data('val', dataset, datadir, nchannels)

    print("trainings set size: ", trainingsetsize)

    # random seed with restricted size
    sampler = RandomSampler(train_dataset, replacement=True, num_samples=trainingsetsize)

    train_loader = DataLoader(train_dataset, batch_size=batchsize, shuffle=False, sampler=sampler)
    val_loader = DataLoader(val_dataset, batch_size=batchsize, shuffle=False)

    for i, (data, target) in enumerate(train_loader):
        print(data.size())
        print(target[:500])

        if random_labels:
            target = target[torch.randperm(target.size()[0])]

        # compute the output
        output = model(data)


if __name__ == '__main__':
    main()
