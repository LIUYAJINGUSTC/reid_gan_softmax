#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 10:45:56 2017

@author: lyj
"""
from __future__ import absolute_import
import torch
from torch import nn
from torch.nn.parameter import Parameter
from reid.utils.serialization import load_checkpoint
from torch import autograd
import torch.optim as optim
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
            #nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(),
            # state size. (ndf*4) x 36 x 14
            nn.Conv2d(self.ndf * 4, self.ndf * 8, kernel_size = (4,4), stride =
                      (2,2), padding = (1,1), bias=True),
            #nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(),
            # state size. (ndf*8) x 18 x 7
            nn.Conv2d(self.ndf * 8, self.ndf * 16, kernel_size = (4,4), stride =
                      (2,2), padding = (1,1), bias=True),
            #nn.BatchNorm2d(ndf * 16),
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
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.01)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def load_params(new_model, pretrained_model):
    new_model_dict = new_model.module.state_dict()
    pretrained_checkpoint = load_checkpoint(pretrained_model)
    #for name, param in pretrained_checkpoint.items():
    for name, param in pretrained_checkpoint.items():
        print('pretrained_model params name and size: ', name, param.size())

        if name in new_model_dict and 'classifier' not in name:
            if isinstance(param, Parameter):
                param = param.data
            try:
                new_model_dict[name].copy_(param)
                print('############# new_model load params name: ',name)
            except:
                raise RuntimeError('While copying the parameter named {}, ' +
                                   'whose dimensions in the model are {} and ' +
                                   'whose dimensions in the checkpoint are {}.'
                                   .format(name, new_model_dict[name].size(), param.size()))
        else:
            continue

def netd(netD_path = '',**kwargs):
    model = _netD(**kwargs)
    model.apply(weights_init)
    if netD_path != '':
        load_params(model,netD_path)
    return model
class DiscriLoss(nn.Module):

    def __init__(self,cuda,lr,batchsize,height,width,step_size,decay_step):
        super(DiscriLoss, self).__init__()

        self.model = netd()
        self.cuda = cuda
        self.batchsize = batchsize
        self.height = height
        self.width = width
        self.lr = lr
        self.step_size = step_size
        self.decay_step = decay_step
        if self.cuda:
            self.model = self.model.cuda()
        self.opti =self._opti()

    def _opti(self):

        optimizerD = optim.Adam(self.model.parameters(), lr=self.lr, betas=(0.5, 0.9))
        return optimizerD
    def calc_gradient_penalty(self,real_data, fake_data):

        alpha = torch.rand(self.batchsize, 1)
        alpha = alpha.expand(self.batchsize, real_data.nelement()/self.batchsize).contiguous().view(self.batchsize, \
                        3, self.height, self.width)
        alpha = alpha.cuda() if self.cuda else alpha

        interpolates = alpha * real_data + ((1 - alpha) * fake_data)

        if self.cuda:
            interpolates = interpolates.cuda()
        interpolates = autograd.Variable(interpolates, requires_grad=True)
        disc_interpolates = self.model(interpolates)

        gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).cuda() if self.cuda else torch.ones(
                                  disc_interpolates.size()),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]

        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * 10
        return gradient_penalty

    def adjust_lr(self,epoch):
        lr = self.lr if epoch <= self.step_size else \
                self.lr * (0.1 ** ((epoch - self.step_size) // self.decay_step + 1))
        for g in self.opti.param_groups:
            g['lr'] = lr * g.get('lr_mult', 1)
        print('epoch: {}, lr: {}'.format(epoch,g['lr']))

    def forward(self, fake, inputv, i):

        self.model.zero_grad()
        output = self.model(inputv)
        D_real = output.mean()
        output = self.model(fake.detach())
        D_fake = output.mean()
        gradient_penalty = self.calc_gradient_penalty(inputv.data, fake.data)
        errD = D_fake - D_real + gradient_penalty
        errD.backward()
        self.opti.step()
        print('Loss_D:%.4f'% (errD.data[0]))

    def gloss(self, fake):

        output = self.model(fake)
        G = output.mean()
        return -G
def discri_loss(**kwargs):
    return DiscriLoss(**kwargs)
