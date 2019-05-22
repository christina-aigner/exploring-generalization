from models.model_utils import load_model, load_checkpoint_dict, reparam
from torch.utils.data import DataLoader
from models import vgg, fc
import numpy as np
from norms.measures import *
import argparse


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

    def validate(args, model, device, data_loader: DataLoader, criterion):
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

            if data_loader.sampler:
                len_dataset = len(data_loader.sampler)
            else:
                len_dataset = len(data_loader.dataset)

        return 1 - (sum_correct / len_dataset), (sum_loss / len_dataset), margin

    def calculate_norms(self, init_model, device, nchannels=3, img_dim=32):
        model = copy.deepcopy(self.model)
        reparam(model)
        reparam(init_model)

        margin = self.checkpoint['margin']

        with torch.no_grad():
            self.l1_norm = calc_measure(model, init_model, norm, 'product',
                                        {'p': 1, 'q': float('Inf')}) / margin
            self.l2_norm = calc_measure(model, init_model, norm, 'product',
                                        {'p': 2, 'q': 2}) / margin
            self.spectral_norm = calc_measure(model, init_model, op_norm,
                                              'product',
                                              {'p': float('Inf')}) / margin
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
        d = calc_measure(model, base_model, depth, 'sum', {})
        # number of parameters (not including batch norm)
        nparam = calc_measure(model, base_model, n_param, 'sum', {})

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

    def calculate_sharpness(self):
        pass

    def calculate_KL(self):
        pass


def main():
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
    args = parser.parse_args()

    # set up
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    nchannels, nclasses, img_dim, = 3, 10, 32

    # create an initial model
    if args.network == 'vgg':
        # customized vgg network
        model = vgg.Network(nchannels, nclasses)
    elif args.network == 'fc':
        # two layer perceptron
        model = fc.Network(nchannels, nclasses)

    model.to(device)

    eval = Evaluation(args.trainingsetsize, args.modelpath)
    base_model = vgg.Network(3, 10)

    eval.calculate_norms(base_model, device)
    print(f"L1 Norm: {eval.l1_norm}")
    print(f"L2 Norm: {eval.l2_norm}")
    print(f"L1-path-norm: {eval.l1_path_norm}")
    print(f"L2-path-norm: {eval.l2_path_norm}")
    print(f"Spectral Norm: {eval.spectral_norm}")

    print(f"Sharpness: todo")


if __name__ == '__main__':
    main()
