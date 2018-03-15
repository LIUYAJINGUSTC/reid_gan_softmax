#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 11:46:47 2017

@author: lyj
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from collections import OrderedDict
from torch.nn import init
import pdb
__all__ = ['DenseNet', 'densenet121_pre']


model_urls = {
    'densenet121': 'https://download.pytorch.org/models/densenet121-a639ec97.pth'
}


def densenet121_pre(pretrained=False,hidden_num=2048, **kwargs):
    r"""Densenet-121 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = DenseNet(num_init_features=64, growth_rate=32, hidden_num=2048,block_config=(6, 12,24,16),
                     **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['densenet121']))
    return model

class _DenseLayer(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, bn_size,drop_rate):
        super(_DenseLayer, self).__init__()
        self.add_module('norm.1', nn.BatchNorm2d(num_input_features)),
        self.add_module('relu.1', nn.ReLU(inplace=True)),
        self.add_module('conv.1', nn.Conv2d(num_input_features, bn_size *
                        growth_rate, kernel_size=1, stride=1, bias=False)),
        self.add_module('norm.2', nn.BatchNorm2d(bn_size * growth_rate)),
        self.add_module('relu.2', nn.ReLU(inplace=True)),
        self.add_module('conv.2', nn.Conv2d(bn_size * growth_rate, growth_rate,
                        kernel_size=3, stride=1, padding=1, bias=False)),
        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = super(_DenseLayer, self).forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return torch.cat([x, new_features], 1)


class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate,drop_rate):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features + i * growth_rate,growth_rate, bn_size, drop_rate)
            self.add_module('denselayer%d' % (i + 1), layer)


class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))


class DenseNet(nn.Module):
    r"""Densenet-BC model class, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_
    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
    """
    def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16),
                 num_init_features=64, bn_size=4, drop_rate=0,
                 hidden_num = 2048,cut_at_pooling=False,
                 num_features=0, norm=False, dropout=0, num_classes=0):

        super(DenseNet, self).__init__()
        self.cut_at_pooling = cut_at_pooling
        # First convolution
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(3, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)),
            ('norm0', nn.BatchNorm2d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool2d(kernel_size=3, stride=2,padding=1)),
        ]))
        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(num_layers=num_layers, num_input_features=num_features,
                                bn_size=bn_size, growth_rate=growth_rate,
                                drop_rate=drop_rate)
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features,
                                    num_output_features=num_features //
                                    2)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2
                if i == 1:
                    feature_count = num_features
        self.features.add_module('norm5', nn.BatchNorm2d(num_features))
#        self.hidden = nn.Sequential(OrderedDict([
#            ('norm2', nn.BatchNorm2d(feature_count)),
#            ('relu2', nn.ReLU(inplace=True)),
#            ('conv3', nn.Conv2d(feature_count, feature_count, kernel_size=3,
#                                stride=1, padding=1, bias=False)),
#            ('norm3', nn.BatchNorm2d(feature_count)),
#            ('relu3', nn.ReLU(inplace=True)),
#            ('pool3', nn.MaxPool2d(kernel_size=2, stride=2, padding=1)),
#            ('conv4', nn.Conv2d(feature_count, feature_count, kernel_size=3,
#                                stride=1, padding=1, bias=False)),
#            ('norm4', nn.BatchNorm2d(feature_count)),
#            ('relu4', nn.ReLU(inplace=True)),
#            ('pool4', nn.MaxPool2d(kernel_size=2, stride=2, padding=1)),
#        ]))
#        self.embedding = nn.Sequential(OrderedDict([
#            ('feat', nn.Linear(3 * 6 * 256,hidden_num)),
#            ('feat_bn', nn.BatchNorm2d(hidden_num)),
#            ('feat_relu',nn.ReLU(inplace=True))
#        ]))
