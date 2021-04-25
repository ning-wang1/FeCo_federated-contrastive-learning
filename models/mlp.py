import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class MLP(nn.Module):
    def __init__(self, input_size=10, layer1_size=32, layer2_size=16, output_size=2):
        super(MLP, self).__init__()

        self.input_size = input_size
        self.layer1_size = layer1_size
        self.layer2_size = layer2_size
        self.output_size = output_size

        self.layer1 = nn.Linear(input_size, layer1_size)
        self.layer2 = nn.Linear(layer1_size, layer2_size)
        self.output = nn.Linear(layer2_size, output_size)

    def forward(self, x):
        # x = F.dropout(self.layer1(x), p=0.1)
        x = self.layer1(x)
        x = F.relu(x)
        x = F.relu(self.layer2(x))
        x = self.output(x)
        x = x.view(x.size(0), -1)
        normed_x = F.normalize(x, p=2, dim=1)

        return x, normed_x


class ProjectionHead(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ProjectionHead, self).__init__()
        self.hidden = nn.Linear(input_dim, 256)
        self.relu = nn.ReLU(inplace=True)
        self.out = nn.Linear(256, output_dim)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight = nn.init.kaiming_normal_(m.weight, mode='fan_out')
                m.bias.data.fill_(0.01)

    def forward(self, x):
        x = self.hidden(x)
        x = self.relu(x)
        x = self.out(x)
        x = F.normalize(x, p=2, dim=1)

        return x


def get_model(**kwargs):
    """
    Returns the model.
    """
    model = MLP(**kwargs)
    return model