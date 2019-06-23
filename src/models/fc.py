import torch.nn as nn

class Network(nn.Module):
    """
    Builds a two layer perceptron as specified in Neyshabur et. al. 2017 Appendix A.
    With dynamic number of hidden units, with given number of input channels
    and fixed number of image input dimensions (suitable for CIFAR and MNIST).
    """

    def __init__(self, hiddenunits, nchannels, nclasses):
        super(Network, self).__init__()
        self.classifier = nn.Sequential(nn.Linear( nchannels * 32 * 32, hiddenunits ), nn.ReLU(inplace=True),
                                        nn.Linear( hiddenunits, nclasses))

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
