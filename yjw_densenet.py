from __future__ import absolute_import

from torch import nn
from torch.nn import functional as F
from torch.nn import init
import torchvision
from collections import OrderedDict
from densenet_ini import *

__all__ = ['DenseNet', 'densenet121', 'densenet169', 'densenet201', 'densenet161']


class DenseNet(nn.Module):

    def __init__(self, depth, pretrained=True, cut_at_pooling=False,
                 num_features=0, norm=False, dropout=0,
                 num_classes=0):
        super(DenseNet, self).__init__()
        self.pretrained = pretrained
        self.cut_at_pooling = cut_at_pooling

        # Construct base (pretrained) densenet
        self.base = densenet121_pre(pretrained=False,num_classes = num_classes)
        if not self.cut_at_pooling:
            self.num_features = num_features
            self.norm = norm
            self.dropout = dropout
            self.has_embedding = num_features > 0
            self.num_classes = num_classes

            out_planes = 1024

            # Append new layers
            if self.has_embedding:
                self.feat = nn.Linear(out_planes, self.num_features)
                self.feat_bn = nn.BatchNorm1d(self.num_features)
                init.kaiming_normal(self.feat.weight, mode='fan_out')
                init.constant(self.feat.bias, 0)
                init.constant(self.feat_bn.weight, 1)
                init.constant(self.feat_bn.bias, 0)
            else:
                # Change the num_features to CNN output channels
                self.num_features = out_planes
            if self.dropout > 0:
                self.drop = nn.Dropout(self.dropout)
            if self.num_classes > 0 :
               # self.classifier_bn = nn.BatchNorm1d(self.num_features)
                self.classifier = nn.Linear(self.num_features, self.num_classes)
               # init.constant(self.classifier_bn.weight, 1)
               # init.constant(self.classifier_bn.bias, 0)
                init.normal(self.classifier.weight, std=0.001)
                init.constant(self.classifier.bias, 0)

        if not self.pretrained:
            self.densenet_params()

    def forward(self, x):
        rec_feature = x
        x = self.base.features(x)
        x = F.relu(x, inplace=True)
        if self.cut_at_pooling:
            return x
        x = F.avg_pool2d(x, x.size()[2:])
        x = x.view(x.size(0), -1)
        if self.has_embedding:
            x = self.feat(x)
            x = self.feat_bn(x)
        if self.norm:
            x = F.normalize(x)
        elif self.has_embedding:
            x_norm = F.normalize(x)
            #print('enter')
            x_class = F.relu(x)

        if self.dropout > 0:
            x_class = self.drop(x_class)
        if self.num_classes > 0:
            #x_class = self.classifier_bn(x_class)
            x_class = self.classifier(x_class)
        for name, module in self.base.features._modules.items():
            if 'conv0' in name or 'pool0' in name or 'norm0' in name or 'relu0' in name:
                rec_feature = module(rec_feature)
                #print(name,rec_feature.size)
            if 'denseblock1' in name or 'denseblock2' in name or 'transition1' in name or 'transition2' in name:
                rec_feature = module(rec_feature)
                #print(name,rec_feature.size)
        return x_class,rec_feature,x_norm
    def densenet_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant(m.weight, 1)
                init.constant(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant(m.bias, 0)


def densenet121(**kwargs):
    return DenseNet(121, **kwargs)


def densenet169(**kwargs):
    return DenseNet(169, **kwargs)


def densenet201(**kwargs):
    return DenseNet(201, **kwargs)


def densenet161(**kwargs):
    return DenseNet(161, **kwargs)
