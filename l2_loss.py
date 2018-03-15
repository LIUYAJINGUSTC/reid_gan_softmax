#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 26 10:37:34 2017

@author: lyj
"""
import torch.nn as nn
import math
import torch

class L2Loss(nn.Module):

    def __init__(self,cuda):
        super(L2Loss, self).__init__()

    def _euclidean_distance(self, x1, x2, p=2, eps=1e-6):
        """ compute enclidean distance of two inputs"""
        assert x1.size() == x2.size()
        diff = torch.abs(x1 - x2)
        return torch.mean(torch.pow(diff + eps, p))
    def forward(self, fake, inputv): 
        l2_loss = self._euclidean_distance(fake, inputv)
        return l2_loss

def l2_loss(**kwargs):
    return L2Loss(**kwargs)
