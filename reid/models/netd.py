#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 10:45:56 2017

@author: lyj
"""
from __future__ import absolute_import
import torch
from torch import nn
class _netD(nn.Module):
    def __init__(self,ngpu = 1,nc = 3,ndf =64):
        super(_netD, self).__init__()
        self.ngpu = ngpu
        self.nc = nc
        self.ndf = ndf
        self.main = nn.Sequential(
            # input is (nc) x 288 x 112
            nn.Conv2d(self.nc, self.ndf, kernel_size = (4,4), stride = (2,2),
                      padding = (1,1), bias=True),
            nn.LeakyReLU(),
            # state size. (ndf) x 144 x 56
            nn.Conv2d(self.ndf, self.ndf * 2, kernel_size = (4,4), stride = (2,2),
                      padding = (1,1), bias=True),
            nn.LeakyReLU(),
            # state size. (ndf * 2) x 72 x 28
            nn.Conv2d(self.ndf * 2, self.ndf * 4, kernel_size = (4,4), stride =
                      (2,2), padding = (1,1), bias=True),
            #nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(),
            # state size. (ndf*4) x 36 x 14
            nn.Conv2d(self.ndf * 4, self.ndf * 8, kernel_size = (4,4), stride =
                      (2,2), padding = (1,1), bias=True),
            #nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(),
            # state size. (ndf*8) x 18 x 7
            nn.Conv2d(self.ndf * 8, self.ndf * 16, kernel_size = (4,4), stride =
                      (2,2), padding = (1,1), bias=True),
            #nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(),
            # state size. (ndf*16) x 9 x 3
            # nn.Conv2d(ndf * 16, 1, kernel_size = (9,3), stride = (1,1), padding = (0,0), bias=False),
            #nn.Sigmoid()
        )
        self.linear = nn.Linear(9*3*16*self.ndf, 1)
    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
            output = output.view(-1,self.ndf * 16 * 9 * 3)
            output = self.linear(output)
        else:
            output = self.main(input)
            output = output.view(-1,self.ndf * 16 * 9 * 3)
            output = self.linear(output)
        return output
def netd(**kwargs):
    return _netD(**kwargs)
