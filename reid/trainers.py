from __future__ import print_function, absolute_import
import time

import torch
from torch.autograd import Variable

from .evaluation_metrics import accuracy
from .loss import OIMLoss, TripletLoss
from .utils.meters import AverageMeter
import pdb

class BaseTrainer(object):
    def __init__(self, model, criterion):
        super(BaseTrainer, self).__init__()
        self.model = model
        self.criterion = criterion

    def train(self, epoch, data_loader, optimizer, print_freq=1, print_info=10):
        self.model.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        precisions = AverageMeter()

        end = time.time()
        for i, inputs in enumerate(data_loader):
            data_time.update(time.time() - end)

            inputs, targets = self._parse_data(inputs)
            loss, prec1 = self._forward(inputs, targets)

            losses.update(loss.data[0], targets.size(0))
            precisions.update(prec1, targets.size(0))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0:
                print('Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      'Loss {:.3f} ({:.3f})\t'
                      'Prec {:.2%} ({:.2%})\t'
                      .format(epoch, i + 1, len(data_loader),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              losses.val, losses.avg,
                              precisions.val, precisions.avg))
        if (epoch+1) % print_info == 0:
            self.show_info()

    def show_info(self, with_arch=True, with_grad=True):
        if with_arch:
            print('\n\n################# modules ###################')
            for name, m in self.model.named_modules():
                print('{}: {}'.format(name, m))
            print('################# modules ###################\n\n')
        if with_grad:
            print('################# params diff ###################')
            for name, param in self.model.named_parameters():
                print(name)
                if 'base.classifier' not in name:
                    mean_value = torch.abs(param.data).mean()
                    mean_grad = torch.abs(param.grad).mean().data[0] + 1e-8
                    print('{}: size{}, data_abd_avg: {}, dgrad_abd_avg: {}, data/grad: {}'.format(name,
                                                param.size(), mean_value, mean_grad, mean_value/mean_grad))
            print('################# params diff ###################\n\n')
        else:
            print('################# params ###################')
            for name, param in self.model.named_parameters():
                print('{}: size{}, abs_avg: {}'.format(name,
                                                       param.size(),
                                                       torch.abs(param.data.cpu()).mean()))
            print('################# params ###################\n\n')
    def _parse_data(self, inputs):
        raise NotImplementedError

    def _forward(self, inputs, targets):
        raise NotImplementedError


class Trainer(BaseTrainer):
    def _parse_data(self, inputs):
        imgs, _, pids, _ = inputs
        inputs = [Variable(imgs)]
        targets = Variable(pids.cuda())
        return inputs, targets

    def _forward(self, inputs, targets):
        outputs, _ = self.model(*inputs)
        if isinstance(self.criterion, torch.nn.CrossEntropyLoss):
            loss = self.criterion(outputs, targets)
            prec, = accuracy(outputs.data, targets.data)
            prec = prec[0]
        elif isinstance(self.criterion, OIMLoss):
            loss, outputs = self.criterion(outputs, targets)
            prec, = accuracy(outputs.data, targets.data)
            prec = prec[0]
        elif isinstance(self.criterion, TripletLoss):
            loss, prec = self.criterion(outputs, targets)
        else:
            raise ValueError("Unsupported loss:", self.criterion)
        return loss, prec
