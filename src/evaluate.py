import argparse

import torch
from torch.nn import CrossEntropyLoss

from models import vgg
from utils.data_utils import CIFARSubset
from utils.eval_utils import calculate
from utils.model_utils import load_checkpoint_dict, load_model
from utils.plot_utils import plot_list

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
    init_model = vgg.Network(3, 10)

    real_labels_all = [
        (1000, 'checkpoint_1000_538.pth'),
        (2000, 'checkpoint_2000_559.pth'),
        (10000, 'checkpoint_10000_434.pth'),
        (20000, 'checkpoint_20000_599.pth'),
        (30000, 'checkpoint_30000_599.pth'),
        (40000, 'checkpoint_40000_599.pth')
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
        (5000, 'checkpoint_5000_999.pth')
    ]

    random_labels_largeset = [
        (10000, 'checkpoint_10000_999.pth'),
        (20000, 'checkpoint_20000_999.pth'),
        (30000, 'checkpoint_30000_999.pth'),
        (40000, 'checkpoint_40000_999.pth'),
        (50000, 'checkpoint_50000_999.pth')
    ]
    # (50000, 'checkpoint_50000_500.pth')

    l1_norms = []
    l2_norms = []
    spec_norms = []
    l1_path_norms = []
    l2_path_norms = []
    l1_max_bounds = []
    frobenius_bounds = []
    spec_l2_bounds = []

    for model in random_labels_smallset:
        # setup - parse checkpoint
        setsize, filename = model
        checkpoint = load_checkpoint_dict(f'../saved_models/random_labels/training_set/' + filename)
        margin: int = checkpoint['margin']
        used_targets = checkpoint['rand_targets']
        model = load_model(f'../saved_models/random_labels/training_set/' + filename)
        train_loader, val_loader = CIFARSubset(args, **kwargs)
        criterion = CrossEntropyLoss().to(device)

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

        sharpness = calc_sharpness(model, init_model, device, train_loader, criterion)

    plot_list(l1_norms, 'l1 norm')
    plot_list(l2_norms, 'l2 norm')
    plot_list(spec_norms, 'spectral norm')
    plot_list(l1_path_norms, 'l1 path norm')
    plot_list(l2_path_norms, 'l2 path norm')

    plot_list(l1_max_bounds, 'l1 max bound')
    plot_list(spec_l2_bounds, 'spectral l2 bound')
    plot_list(frobenius_bounds, 'frobenius bound')
