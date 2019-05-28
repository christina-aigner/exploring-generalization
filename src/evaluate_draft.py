import argparse

from measures.measures import *
from models import vgg
from utils.model_utils import load_model, load_checkpoint_dict, reparam


def calc_measure(model, init_model, measure_func, operator, kwargs={}, p=1):
    measure_val = 0
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


class Evaluation:
    def __init__(self, trainingsetsize, checkpoint_path):
        self.checkpoint = load_checkpoint_dict(checkpoint_path)
        self.trainingsetsize = trainingsetsize
        self.hidden = 1
        self.model = load_model(checkpoint_path)
        self.l1_norm = None
        self.l2_norm = None
        self.l1_path_norm = None
        self.l2_path_norm = None
        self.spectral_norm = None
        self.sharpness = None

    def calculate_norms(self, init_model, device, nchannels=3, img_dim=32):
        model = copy.deepcopy(self.model)
        reparam(model)
        reparam(init_model)

        margin: int = self.checkpoint['margin']

        print(f"depth {get_depth(model)}")
        print(f"params {get_npara(model)}")
        model_list = list(model.children())

        with torch.no_grad():
            self.l1_norm = calc_norm(model_list, 1, float('Inf'), 0) / margin
            self.l2_norm = calc_norm(model, 2, 2, 0) / margin
            self.spectral_norm = calc_spectral_norm(model, float('Inf'), 0) / margin
            self.l1_path_norm = lp_path_norm(model, device, p=1,
                                             input_size=[1, nchannels, img_dim,
                                                         img_dim]) / margin
            self.l2_path_norm = lp_path_norm(model, device, p=2,
                                             input_size=[1, nchannels, img_dim,
                                                         img_dim]) / margin

    def calculate_bounds(self, measure, base_model, device, nchannels=3, img_dim=32):
        """
        Generalization bounds: constants and additive logarithmic factors are not included
        This value of alpha is based on the improved depth dependency by Golowith et al. 2018
        Args:
            base_model:
            device:
            nchannels:
            img_dim:

        Returns:

        """

        bound = {}
        model = copy.deepcopy(self.model)
        reparam(model)
        # depth
        d = get_depth(model, base_model)
        # number of parameters (not including batch norm)
        nparam = get_npara(model, base_model)

        alpha = math.sqrt(d + math.log(nchannels * img_dim * img_dim))

        bound['L1_max Bound (Bartlett and Mendelson 2002)'] = alpha * measure[
            'L_{1,inf} norm'] / math.sqrt(self.trainingsetsize)
        bound['Frobenious Bound (Neyshabur et al. 2015)'] = alpha * measure[
            'Frobenious norm'] / math.sqrt(self.trainingsetsize)

        beta = math.log(self.trainingsetsize) * math.log(nparam)
        ratio = calc_measure(model, base_model, h_dist_op_norm, 'norm',
                             {'p': 2, 'q': 1, 'p_op': float('Inf')}, p=2 / 3)
        bound['Spec_L_{2,1} Bound (Bartlett et al. 2017)'] = beta * measure[
            'Spectral norm'] * ratio / math.sqrt(self.trainingsetsize)

        ratio = calc_measure(model, base_model, h_dist_op_norm, 'norm',
                             {'p': 2, 'q': 2, 'p_op': float('Inf')}, p=2)
        bound['Spec_Fro Bound (Neyshabur et al. 2018)'] = d * measure[
            'Spectral norm'] * ratio / math.sqrt(self.trainingsetsize)

        return bound

    def calc_sharpness(model, init_model, device, train_loader, criterion):
        clean_model = copy.deepcopy(model)
        clean_error, clean_loss, clean_margin = validate(clean_model, device, train_loader,
                                                         criterion)
        calc_measure(model, init_model, add_gauss_perturbation, 'sharpness')
        pert_error, pert_loss, pert_margin = validate(model, device, train_loader, criterion)
        return pert_loss - clean_loss

    def calculate_KL(self):
        pass


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


def PAC_KL(tr_loss, exp_sharpness, l2_reg, setsize, sigma=1, delta=0.2):
    """

    Args:
        tr_loss: training loss
        exp_sharpness: expected sharpness
        l2_reg: l2 regularization of the model = |w|2
        setsize: training set size of the training data
        sigma: guassian variance
        delta: probability, 1-delta is the prob. over the draw of the training set

    Returns:

    """
    term = 4 * math.sqrt(
        ((1 / setsize) * (l2_reg / (2 * (sigma ^ 2)))) + math.log((2 * setsize) / delta))
    return tr_loss + exp_sharpness + term


if __name__ == '__main__':
    main()
