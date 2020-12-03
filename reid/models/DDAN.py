from __future__ import absolute_import

from torch import nn
from torch.nn import functional as F
from torch.nn import init
import torchvision
import pdb
from . import mobilenet as mobilenet


class Encoder(nn.Module):
    def __init__(self, pretrained=True, cut_at_pooling=False,
                 num_features=0, norm=False, dropout=0, num_classes=0, triplet_features=0):
        super(Encoder, self).__init__()
        self.base = mobilenet.mobilenet_v2(pretrained=True)

    def forward(self, x, output_feature=None):    
        x = self.base(x,'encoder')

        return x

class TransferNet(nn.Module):
    def __init__(self, pretrained=True, cut_at_pooling=False,
                 num_features=0, norm=False, dropout=0, num_classes=0, triplet_features=0):
        super(TransferNet, self).__init__()

        self.base = mobilenet.mobilenet_v2(pretrained=True)

        self.dropout = dropout
        self.num_classes = num_classes
        out_planes = 1280
        self.bn = nn.BatchNorm1d(out_planes)
        init.constant_(self.bn.weight, 1)
        init.constant_(self.bn.bias, 0)

        self.drop = nn.Dropout(self.dropout)
        self.classifier = nn.Linear(out_planes, self.num_classes)


    def forward(self, x, output_feature=None):
        x = self.base(x,'task')
        x = F.avg_pool2d(x, x.size()[2:])
        x = x.view(x.size(0), -1)

        x_feature = x
        x = self.bn(x)
        if output_feature == 'pool5':
            return x
        x = self.drop(x)
        x_class = self.classifier(x)
        return x_class, x_feature

class CamDiscriminator(nn.Module):
    def __init__(self, channels=1280):
        super(CamDiscriminator, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        out_planes = channels
        self.feat = nn.Linear(out_planes, 128)
        self.bn = nn.BatchNorm1d(128)
        init.constant_(self.bn.weight, 1)
        init.constant_(self.bn.bias, 0)
        self.num_classes = 2
        self.classifier = nn.Linear(128, self.num_classes)

    def forward(self, x):
        x = self.feat(x)
        x = self.bn(x)
        self.drop = nn.Dropout(0.5)
        x = self.classifier(x)
        return x

def DDAN(**kwargs):
    return Encoder(50, **kwargs), TransferNet(50, **kwargs), CamDiscriminator(channels=1280)
