import torch.nn as nn

class Network(nn.Module):
    def __init__(self, nchannels, nclasses):
        super(Network, self).__init__()
        config = [64, 64, 'MaxPool', 128, 128, 'MaxPool', 256, 256, 'MaxPool', 512, 512, 'AvgPool']
        self.features = generate_layers(config, nchannels)
        self.classifier = nn.Sequential( nn.Linear( 512, 512 ), nn.BatchNorm1d(512),
                                        nn.ReLU(inplace=True), nn.Dropout(0.5), nn.Linear( 512, nclasses))

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def generate_layers(config, in_channels):
    layers = []
    for in_ in config:
        if in_ == 'MaxPool':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2), nn.Dropout(0.5)]
        elif in_ == 'AvgPool':
            layers += [nn.AvgPool2d(kernel_size=4, stride=4), nn.Dropout(0.5)]
        else: #nn.BatchNorm2d(v)
            layers += [nn.Conv2d(in_channels, in_, kernel_size=3, padding=1), nn.ReLU(inplace=True)]
            in_channels = in_
    return nn.Sequential(*layers)
