import argparse

import torch
from torch.nn import CrossEntropyLoss

from models import fc
from utils.data_utils import MNIST
from utils.eval_utils import calculate
from utils.model_utils import load_checkpoint_dict, load_model
from utils.plot_utils import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluation of a pre-trained model')
    # neccessary argument
    parser.add_argument('--modelpath', type=str,
                        help='path from which the pre-trained model should be loaded')
    # optional
    parser.add_argument('--model', default='vgg', type=str,
                        help='model type that should be evaluated, (options: vgg | fc, default= vgg)')
    parser.add_argument('--trainingsetsize', default=50000, type=int,
                        help='size of the training set of the loaded model, (default: 50,000)')
    parser.add_argument('--datadir', default='../datasets', type=str,
                        help='path to the directory that contains the datasets')
    parser.add_argument('--randomlabels', default=False, type=bool,
                        help='training with random labels Yes or No? (options: True | False, default: False)')

    args = parser.parse_args()

    # set up
    use_cuda = torch.cuda.is_available()
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    nchannels, nclasses, img_dim, = 3, 10, 32

    real_labels_all = [
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

    random_labels_all = [
        (1000, 'checkpoint_1000_999.pth'),
        (2000, 'checkpoint_2000_999.pth'),
        (3000, 'checkpoint_3000_999.pth'),
        (4000, 'checkpoint_4000_999.pth'),
        # (5000, 'checkpoint_5000_999.pth'),
        (10000, 'checkpoint_10000_999.pth'),
        (20000, 'checkpoint_20000_999.pth'),
        # (30000, 'checkpoint_30000_999.pth'),
        # (40000, 'checkpoint_40000_999.pth'),
        # (50000, 'checkpoint_50000_999.pth')
    ]

    real_labels_smallset = [
        (1000, 'checkpoint_1000_538.pth'),
        (2000, 'checkpoint_2000_559.pth'),
        (3000, 'checkpoint_3000_463.pth'),
        (4000, 'checkpoint_4000_468.pth'),
        (5000, 'checkpoint_5000_504.pth')
    ]

    real_labels_largeset = [
        (10000, 'checkpoint_10000_434.pth'),
        (20000, 'checkpoint_20000_599.pth'),
        (30000, 'checkpoint_30000_599.pth'),
        (40000, 'checkpoint_40000_599.pth'),
        (50000, 'checkpoint_50000_999.pth')
    ]

    random_labels_smallset = [
        (1000, 'checkpoint_1000_999.pth'),
        (2000, 'checkpoint_2000_999.pth'),
        (3000, 'checkpoint_3000_999.pth'),
        (4000, 'checkpoint_4000_999.pth'),
        #(5000, 'checkpoint_5000_999.pth')
    ]

    random_labels_largeset = [
        (10000, 'checkpoint_10000_999.pth'),
        (20000, 'checkpoint_20000_999.pth'),
        (30000, 'checkpoint_30000_999.pth'),
        (40000, 'checkpoint_40000_999.pth'),
        (50000, 'checkpoint_50000_999.pth')
    ]

    fc_random = [
        (10000, 'cp_fc_8_10000_1499.pth', 8),
        (10000, 'cp_fc_16_10000_1000.pth', 16),
        (10000, 'cp_fc_32_10000_799.pth', 32),
        (10000, 'cp_fc_64_10000_397.pth', 64),
        (10000, 'cp_fc_128_10000_104.pth', 128),
        (10000, 'cp_fc_256_10000_30.pth', 256),
        (10000, 'cp_fc_512_10000_15.pth', 512),
        (10000, 'cp_fc_1024_10000_20.pth', 1024),
        (10000, 'cp_fc_2048_10000_18.pth', 2048)
    ]

    fc_real = [
        (10000, 'cp_fc_8_10000_1499.pth', 8),
        (10000, 'cp_fc_16_10000_889.pth', 16),
        (10000, 'cp_fc_32_10000_201.pth', 32),
        (10000, 'cp_fc_64_10000_199.pth', 64),
        (10000, 'cp_fc_128_10000_77.pth', 128),
        (10000, 'cp_fc_256_10000_25.pth', 256),
        (10000, 'cp_fc_512_10000_15.pth', 512),
        (10000, 'cp_fc_1024_10000_799.pth', 1024)
    ]

    fc_mnist = [
        (60000, 'cp_fc_8_60000_999.pth', 8),
        (60000, 'cp_fc_16_60000_999.pth', 16),
        (60000, 'cp_fc_32_60000_37.pth', 32),
        (60000, 'cp_fc_64_60000_11.pth', 64),
        (60000, 'cp_fc_128_60000_7.pth', 128),
        (60000, 'cp_fc_256_60000_6.pth', 256),
        (60000, 'cp_fc_512_60000_5.pth', 512),
        (60000, 'cp_fc_1024_60000_4.pth', 1024),
        (60000, 'cp_fc_2048_60000_4.pth', 2048),
        (60000, 'cp_fc_4096_60000_4.pth', 4096),
        (60000, 'cp_fc_8192_60000_3.pth', 8192)
    ]

    l1_norms = []
    l2_norms = []
    spec_norms = []
    l1_path_norms = []
    l2_path_norms = []
    l1_max_bounds = []
    frobenius_bounds = []
    spec_l2_bounds = []
    sharpness_list = []
    tr_error_list = []
    tr_loss_list = []
    val_error_list = []

    for model in fc_mnist:
        # setup - parse checkpoint
        path = '../saved_models/real_labels/fc_mnist/'
        setsize, filename, num_hidden = model
        print(num_hidden)
        nchannels = 1
        # init_model = vgg.Network(3, 10)
        init_model = fc.Network(num_hidden, 1, 10)
        checkpoint = load_checkpoint_dict(path + filename)
        margin: int = checkpoint['margin']
        # used_targets = checkpoint['rand_targets']
        model = load_model(path + filename, 'fc', num_hidden, nchannels=1)
        #model = load_model(path + filename)
        #train_labels = checkpoint['rand_targets']
        train_loader, val_loader = MNIST(args, **kwargs)
        #train_loader.dataset.targets = train_labels
        criterion = CrossEntropyLoss().to(device)

        tr_error_list.append(checkpoint['tr_error'])
        tr_loss_list.append(checkpoint['tr_loss'])
        val_error_list.append(checkpoint['val_error'])


        l1, l2, spec, l1_path, l2_path, l1_max_bound, frob_bound, spec_l2_bound = calculate(
            model, init_model, device, setsize, margin, nchannels, nclasses, img_dim)
        l1_norms.append(float(l1))
        l2_norms.append(float(l2))
        spec_norms.append(float(spec))
        l1_path_norms.append(float(l1_path))
        l2_path_norms.append(float(l2_path))
        l1_max_bounds.append(float(l1_max_bound))
        frobenius_bounds.append(float(frob_bound))
        spec_l2_bounds.append(float(spec_l2_bound))

        # sharpness = calc_exp_sharpness(model, init_model, device, train_loader, criterion)
        #print('sharpness', sharpness)

    # x_big = ['10K', '20K', '30K', '40K', '50K']
    x_small = ['1K', '2K', '3K', '4K', '5K', '10K', '20K']
    # x_small = ['10K', '20K', '30K', '50K']
    x_fc = ['8', '16', '32', '64', '128', '256', '512', '1k', '2k', '4k', '8k']
    plot_error(tr_error_list, val_error_list, x_fc, 'Real Labels FC')
    all_data = (l2_norms, l1_path_norms, l2_path_norms, spec_norms)
    plot_all(all_data, x_fc, 'Norms for real labels FC')

    plot_list(l2_norms, 'l2 norm')
    plot_list(l1_path_norms, 'l1path')
    plot_list(l2_path_norms, 'l2path')
    plot_list(spec_norms, 'spectral')
    plot_l2_norm(l2_norms, x_fc)
    print(l2_norms)
    plot_spectral(spec_norms, x_fc)
    print(spec_norms)
    plot_l1_path(l1_path_norms, x_fc)
    print(l1_path_norms)
    plot_l2_path(l2_path_norms, x_fc)
    print(l2_path_norms)

    plot_list(l1_max_bounds, 'l1 max bound')
    plot_list(spec_l2_bounds, 'spectral l2 bound')
    plot_list(frobenius_bounds, 'frobenius bound')
    #plot_list(sharpness_list, 'sharpness')
