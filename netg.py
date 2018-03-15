#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 10:45:55 2017

@author: lyj
"""
from __future__ import absolute_import
import torch
from torch import nn
from collections import OrderedDict
class _netG(nn.Module):

    def __init__(self,ngpu = 1,nz = 2048,ngf = 64,nc = 3,feature_count = 256):
        super(_netG, self).__init__()
        self.ngpu = ngpu
        self.nz = nz
        self.ngf = ngf
        self.nc = nc
        self.feature_count = feature_count
        self.hidden = nn.Sequential(OrderedDict([
            ('norm2', nn.BatchNorm2d(self.feature_count)),
            ('relu2', nn.ReLU(inplace=True)),
            ('conv3', nn.Conv2d(self.feature_count, self.feature_count, kernel_size=3,
                                stride=1, padding=1, bias=False)),
            ('norm3', nn.BatchNorm2d(self.feature_count)),
            ('relu3', nn.ReLU(inplace=True)),
            ('pool3', nn.MaxPool2d(kernel_size=2, stride=2, padding=1)),
            ('conv4', nn.Conv2d(self.feature_count, self.feature_count, kernel_size=3,
                                stride=1, padding=1, bias=False)),
            ('norm4', nn.BatchNorm2d(self.feature_count)),
            ('relu4', nn.ReLU(inplace=True)),
            ('pool4', nn.MaxPool2d(kernel_size=2, stride=2, padding=1)),
        ]))
        self.embedding = nn.Sequential(OrderedDict([
            ('feat', nn.Linear(3 * 6 * 256,self.nz)),
           # ('feat_bn', nn.BatchNorm2d(hidden_num)),
           # ('feat_relu',nn.ReLU(inplace=True))
        ]))
        self.preprocess = nn.Sequential(
            nn.Linear(self.nz, self.ngf * 16 * 9 * 3),
            nn.BatchNorm1d(self.ngf * 16 * 9 * 3),
            nn.ReLU(True),
        )
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            # state size. (ngf*8) x 9 x 3
            nn.ConvTranspose2d(self.ngf * 16, self.ngf * 8, kernel_size = (4,4), stride = (2,2), padding = (1,1),output_padding=(0,1) , bias=False),
            nn.BatchNorm2d(self.ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*4) x 18 x 7
            nn.ConvTranspose2d(self.ngf * 8, self.ngf * 4, kernel_size = (4,4), stride = (2,2), padding = (1,1),output_padding=(0,0) , bias=False),
            nn.BatchNorm2d(self.ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*2) x 36 x 14
            nn.ConvTranspose2d(self.ngf * 4, self.ngf * 2, kernel_size = (4,4), stride = (2,2), padding = (1,1),output_padding=(0,0) , bias=False),
            nn.BatchNorm2d(self.ngf * 2),
            nn.ReLU(True),
            # state size. (ngf) x 72 x 28
            nn.ConvTranspose2d(self.ngf * 2, self.ngf , kernel_size = (4,4), stride = (2,2), padding = (1,1),output_padding=(0,0) , bias=False),
            nn.BatchNorm2d(self.ngf),
            nn.ReLU(True),
            # state size. (ngf) x 144 x 56
            nn.ConvTranspose2d(self.ngf,self.nc, kernel_size = (4,4), stride = (2,2), padding = (1,1),output_padding=(0,0) , bias=False),
            nn.Tanh()
            # state size. (nc) x 288 x 112
        )

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            hidden = nn.parallel.data_parallel(self.hidden, input, range(self.ngpu))
            hidden = hidden.view(hidden.size(0),-1)
            feature_new = nn.parallel.data_parallel(self.embedding, hidden, range(self.ngpu))
            output = nn.parallel.data_parallel(self.preprocess, feature_new, range(self.ngpu))
            output = output.view(-1, 16 * self.ngf, 9, 3)
            output = nn.parallel.data_parallel(self.main, output, range(self.ngpu))
        else:
            hidden = self.hidden(input)
            hidden = hidden.view(hidden.size(0),-1) 
            feature_new = self.embedding(hidden)
            output = self.preprocess(feature_new)
            output = output.view(-1, 16 * self.ngf, 9, 3)
            output = self.main(output)
        return output
def netg(**kwargs):
    return _netG(**kwargs)
