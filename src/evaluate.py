import argparse

import torch
from torch.nn import CrossEntropyLoss

from utils.data_utils import MNIST, CIFARSubset
from utils.eval_utils import calculate_norms
from utils.model_utils import load_checkpoint_dict, load_model
from utils.plot_utils import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluation of a pre-trained model')

    parser.add_argument('--modelpath', type=str,
                        help='path from which the pre-trained model should be loaded')
    parser.add_argument('--savedmodels', type=str, default='fc',
                        help='use pretrained models for analysis. Options: vgg | fc')
    # optional
    parser.add_argument('--network', default='vgg', type=str,
                        help='model type that should be evaluated, (options: vgg | fc, default= vgg)')
    parser.add_argument('--trainingsetsize', default=50000, type=int,
                        help='size of the training set of the loaded model, (default: 50,000)')
    parser.add_argument('--datadir', default='../datasets', type=str,
                        help='path to the directory that contains the datasets')
    parser.add_argument('--randomlabels', default=False, type=bool,
                        help='training with random labels Yes or No? (options: True | False, default: False)')
    parser.add_argument('--dataset', default=None,
                        help='load data set for evaluation if sharpness should be calculated, for norms, '
                             'this is not necessary (options: MNIST | CIFARSubset')

    args = parser.parse_args()

    # set up
    use_cuda = torch.cuda.is_available()
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    nclasses, img_dim, = 10, 32

    vgg_models = [
        (1000, 'checkpoint_1000_538.pth'),
        (2000, 'checkpoint_2000_559.pth'),
        (3000, 'checkpoint_3000_463.pth'),
        (4000, 'checkpoint_4000_468.pth'),
        (5000, 'checkpoint_5000_504.pth'),
        (10000, 'checkpoint_10000_434.pth'),
        (20000, 'checkpoint_20000_599.pth'),
        (30000, 'checkpoint_30000_599.pth'),
        (40000, 'checkpoint_40000_599.pth'),
        (50000, 'checkpoint_50000_999.pth')
    ]

    fc_models = [
        (60000, 'cp_fc_8_60000_999.pth', 8),
        (60000, 'cp_fc_16_60000_999.pth', 16),
        (60000, 'cp_fc_32_60000_81.pth', 32),
        (60000, 'cp_fc_64_60000_25.pth', 64),
        (60000, 'cp_fc_128_60000_15.pth', 128),
        (60000, 'cp_fc_256_60000_13.pth', 256),
        (60000, 'cp_fc_512_60000_12.pth', 512),
        (60000, 'cp_fc_1024_60000_10.pth', 1024),
        (60000, 'cp_fc_2048_60000_10.pth', 2048),
        (60000, 'cp_fc_4096_60000_8.pth', 4096),
        (60000, 'cp_fc_8192_60000_7.pth', 8192)
    ]

    # collectors for norms
    l2_norms = []
    spec_norms = []
    l1_path_norms = []
    l2_path_norms = []
    sharpness_list = []
    # collectors for errors and loss
    tr_error_list = []
    tr_loss_list = []
    val_error_list = []

    if args.savedmodels == 'vgg':
        model_list = vgg_models
        path = '../saved_models/real_labels/vgg_cifar/'
        nchannels = 3
        x_error = ['1k', '2k', '3k', '4k', '5k', '10k', '20k', '30K', '40K', '50K']
        x_norm = ['1k', '2k', '3k', '4k', '5k', '10k', '20k', '30K', '40K', '50K']
        xtitle = 'size of the training set'
        error_title = 'Training VGGs on CIFAR'
        norms_title = 'Norms of VGGs trained on CIFAR'
    elif args.savedmodels == 'fc':
        model_list = fc_models
        path = '../saved_models/real_labels/fc_mnist/'
        nchannels = 1
        x_error = ['8', '16', '32', '64', '128', '256', '512', '1k', '2k', '4k', '8k']
        x_norm = ['8', '16', '32', '64', '128', '256', '512', '1k', '2k', '4k', '8k']
        xtitle = '# hidden units'
        error_title = 'Training FCs on MNIST'
        norms_title = 'Norms of FCs trained on MNIST'

    if args.savedmodels == 'vgg' or args.savedmodels == 'fc':
        for i, model in enumerate(model_list):
            # setup - parse checkpoint
            if args.savedmodels == 'vgg':
                setsize, filename = model
                model, init_model = load_model(path + filename)
            elif args.savedmodels == 'fc':
                setsize, filename, num_hidden = model
                model, init_model = load_model(path + filename, 'fc', num_hidden, nchannels)

            checkpoint = load_checkpoint_dict(path + filename)
            margin: int = checkpoint['margin']

            # load same random labels as used in training
            if args.dataset:
                if args.dataset == 'CIFARSubset':
                    train_loader, val_loader = CIFARSubset(args, **kwargs)
                elif args.dataset == 'MNIST':
                    train_loader, val_loader = MNIST(args, **kwargs)
                else:
                    raise KeyError('not a valid argument for dataset')

                if args.randomlabels == True:
                    train_labels = checkpoint['rand_targets']
                    train_loader.dataset.targets = train_labels
                criterion = CrossEntropyLoss().to(device)

            tr_error_list.append(checkpoint['tr_error'])
            tr_loss_list.append(checkpoint['tr_loss'])
            val_error_list.append(checkpoint['val_error'])

            if (checkpoint['tr_error'] < 0.01):
                l2, spec, l1_path, l2_path = calculate_norms(
                    model, device, margin, nchannels, img_dim)
                l2_norms.append(float(l2))
                spec_norms.append(float(spec))
                l1_path_norms.append(float(l1_path))
                l2_path_norms.append(float(l2_path))
            else:
                x_norm.pop(i)

    else:
        raise ValueError('not a valid argument for savedmodels')

    all_data = (l2_norms, l1_path_norms, l2_path_norms, spec_norms)

    plot_error(tr_error_list, val_error_list, x_error, xtitle, error_title)
    plot_normalized(all_data, x_norm, xtitle, norms_title)

    # plot_list(l2_norms, 'l2 norm', x_norm)
    # plot_list(l1_path_norms, 'l1-path norm', x_norm)
    # plot_list(l2_path_norms, 'l2-path norm', x_norm)
    # plot_list(spec_norms, 'spectral norm', x_norm)
