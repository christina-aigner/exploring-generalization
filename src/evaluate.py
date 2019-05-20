from models.model_utils import load_model
from models import vgg
from norms.measures import *
import argparse


class Evaluation:
    def __init__(self, trainingsetsize, checkpoint_path):
        self.trainingsetsize = trainingsetsize
        self.hidden = 1
        self.model = load_model(checkpoint_path)
        self.l2_norm = None
        self.l1_path_norm = None
        self.l2_path_norm = None
        self.spectral_norm = None
        self.sharpness = None
        self.KL = None

    def calculate_norms(self, init_model, device, margin, nchannels=3, img_dim=32):
        model = copy.deepcopy(self.model)
        reparam(model)
        reparam(init_model)

        # size of the training set
        m = self.trainingsetsize

        # depth
        d = calc_measure(model, init_model, depth, 'sum', {})

        # number of parameters (not including batch norm)
        nparam = calc_measure(model, init_model, n_param, 'sum', {})

        measure, bound = {}, {}
        with torch.no_grad():
            measure['L_{1,inf} norm'] = calc_measure(model, init_model, norm, 'product',
                                                     {'p': 1,
                                                      'q': float('Inf')}) / margin
            measure['Frobenious norm'] = calc_measure(model, init_model, norm,
                                                      'product',
                                                      {'p': 2, 'q': 2}) / margin
            measure['L_{3,1.5} norm'] = calc_measure(model, init_model, norm, 'product',
                                                     {'p': 3, 'q': 1.5}) / margin
            measure['Spectral norm'] = calc_measure(model, init_model, op_norm,
                                                    'product',
                                                    {'p': float('Inf')}) / margin
            measure['L_1.5 operator norm'] = calc_measure(model, init_model, op_norm,
                                                          'product',
                                                          {'p': 1.5}) / margin
            measure['Trace norm'] = calc_measure(model, init_model, op_norm, 'product',
                                                 {'p': 1}) / margin
            measure['L1_path norm'] = lp_path_norm(model, device, p=1,
                                                   input_size=[1, nchannels, img_dim,
                                                               img_dim]) / margin
            measure['L1.5_path norm'] = lp_path_norm(model, device, p=1.5,
                                                     input_size=[1, nchannels, img_dim,
                                                                 img_dim]) / margin
            measure['L2_path norm'] = lp_path_norm(model, device, p=2,
                                                   input_size=[1, nchannels, img_dim,
                                                               img_dim]) / margin

            # Generalization bounds: constants and additive logarithmic factors are not included
            # This value of alpha is based on the improved depth dependency by Golowith et al. 2018
            alpha = math.sqrt(d + math.log(nchannels * img_dim * img_dim))

            bound['L1_max Bound (Bartlett and Mendelson 2002)'] = alpha * measure[
                'L_{1,inf} norm'] / math.sqrt(m)
            bound['Frobenious Bound (Neyshabur et al. 2015)'] = alpha * measure[
                'Frobenious norm'] / math.sqrt(m)
            bound['L_{3,1.5} Bound (Neyshabur et al. 2015)'] = alpha * measure[
                'L_{3,1.5} norm'] / (m ** (1 / 3))

            beta = math.log(m) * math.log(nparam)
            ratio = calc_measure(model, init_model, h_dist_op_norm, 'norm',
                                 {'p': 2, 'q': 1, 'p_op': float('Inf')}, p=2 / 3)
            bound['Spec_L_{2,1} Bound (Bartlett et al. 2017)'] = beta * measure[
                'Spectral norm'] * ratio / math.sqrt(m)

            ratio = calc_measure(model, init_model, h_dist_op_norm, 'norm',
                                 {'p': 2, 'q': 2, 'p_op': float('Inf')}, p=2)
            bound['Spec_Fro Bound (Neyshabur et al. 2018)'] = d * measure[
                'Spectral norm'] * ratio / math.sqrt(m)

        return measure, bound

    def calculate_bounds(self):
        pass

    def calculate_sharpness(self):
        pass

    def calculate_KL(self):
        pass


parser = argparse.ArgumentParser(description='Evaluation of a pre-trained model')

parser.add_argument('--trainingsetsize', type=int,
                    help='size of the training set of the loaded model')
parser.add_argument('--modelpath', type=str,
                    help='path from which the pre-trained model should be loaded')
parser.add_argument('--datadir', default='../datasets', type=str,
                    help='path to the directory that contains the datasets')
parser.add_argument('--dataset', default='CIFAR10', type=str,
                    help='name of the dataset (options: MNIST | CIFAR10 | CIFAR100 '
                         '| SVHN, default: CIFAR10)')
args = parser.parse_args()

# set up
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

eval = Evaluation(args.trainingsetsize, args.modelpath)
init_model = vgg.Network(3, 10)

norms, bounds = eval.calculate_norms(init_model, device, 4.496)
print(norms)
