from __future__ import print_function, absolute_import
import argparse
import os.path as osp

import numpy as np
import sys
import torch
from torch.utils.data import DataLoader
from torch.nn.parameter import Parameter
from torch.utils.data.sampler import RandomSampler
import torch.nn as nn
import math
from torch.nn import functional as F
from torch.autograd import Variable
from torch.nn import init
from collections import defaultdict
#from branch import *
#from densenet import *
from yjw_densenet import *
from gan_loss import *
from netg import *
from perceptionloss import perception_loss
from l2_loss import *
import os
import random
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch import autograd
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from reid.utils.data import transforms as T
from reid.utils.data.preprocessor import Preprocessor
from reid.evaluation_metrics import accuracy
from reid.utils.meters import AverageMeter
from PIL import Image
from reid import datasets
#from reid import models
from reid.dist_metric import DistanceMetric
#from reid.trainers import Trainer
from reid.evaluators import Evaluator
from reid.utils.logging import Logger
from reid.utils.serialization import load_checkpoint, save_checkpoint
import pdb

parser = argparse.ArgumentParser(description="Softmax loss classification")
    # data
#parser.add_argument('-d', '--dataset', type=str, default='presons-1115')
parser.add_argument('-b', '--batchSize', type=int, default=128)
parser.add_argument('-j', '--workers', type=int, default=1)
parser.add_argument('--split', type=int, default=0)
parser.add_argument('--height', type=int,default=288,help="input height, default: 256 for resnet*, " \
                    "144 for inception, 288 for densenet")
parser.add_argument('--width', type=int,default=112,
                        help="input width, default: 128 for resnet*, "
                             "56 for inception, 112 for densenet")
parser.add_argument('--combine-trainval', action='store_true',
                        help="train and val sets together for training, "
                             "val set alone for validation")
    # model
parser.add_argument('-a', '--arch', type=str, default='densenet121')
parser.add_argument('--features', type=int, default=256)
parser.add_argument('--dropout', type=float, default=0)
    # optimizer
parser.add_argument('--lr', type=float, default=0.1,
                        help="learning rate of new parameters, for pretrained "
                             "parameters it is 10 times smaller than this")
parser.add_argument('--lr_un', type=float, default=0.001,
                        help="learning rate of new parameters, for pretrained "
                             "parameters it is 10 times smaller than this")
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--weight-decay', type=float, default=5e-4)
parser.add_argument('--step_size', type=int, default=50)
parser.add_argument('--decay_step', type=int, default=30)
parser.add_argument('--LAMBDA', type=int, default=10)
parser.add_argument('--CRITIC_ITERS', type=int, default=1)
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam.default=0.5')
    # adjust lr method
parser.add_argument('--spec-lr-mult', type=float, default=1,help='lr mult of added parameters of spec modules')
parser.add_argument('--spec-params', nargs='+', default=None,help='specified param names')
parser.add_argument('--base-lr-mult', type=float, default=0.1,help='lr mult of added parameters of base modules')
parser.add_argument('--base-params', nargs='+',default=None, help='base param names')
    # training configs  pretrained_model
parser.add_argument('--pretrained_model', type=str, default='', metavar='PATH')
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--netD', default='', help="path to netD (to continue training)")
parser.add_argument('--resume', type=str, default='', metavar='PATH')
parser.add_argument('--resume_epoch', type=int,default=0)
parser.add_argument('--evaluate', action='store_true',
                        help="evaluation only")
parser.add_argument('--epochs', type=int, default=110)
parser.add_argument('--start_save', type=int, default=0,
                        help="start saving checkpoints after specific epoch")
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--print-freq', type=int, default=1)
parser.add_argument('--print-info', type=int, default=10)
parser.add_argument('--cuda', action='store_true', help='enables cuda')
# metric learning
parser.add_argument('--dist-metric', type=str, default='euclidean',
                        choices=['euclidean', 'kissme'])
    # misc
working_dir = osp.dirname(osp.abspath(__file__))
parser.add_argument('--data-dir', type=str, metavar='PATH',
                        default='/home/shenxu.sx/dataset/persons-1129/')
parser.add_argument('--logs-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'logs'))
parser.add_argument('--outf', default='.', help='folder to output images and model checkpoints')
opt = parser.parse_args()
print(opt)


try:
    os.makedirs(opt.outf)
except OSError:
    pass


def get_data(split_id, data_dir, height, width, batchSize, workers, combine_trainval, train_list, \
             val_list, query_list, gallery_list):
    root = data_dir


    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])  # RGB imagenet

    train_set = train_list + val_list if combine_trainval else train_list  # a list

    train_transformer = T.Compose([
        T.RectScale(height, width),
        T.RandomHorizontalFlip(),
        T.ToTensor(),   # [0, 255] to [0.0, 1.0]
        normalizer,     #  normalize each channel of the input
    ])

    test_transformer = T.Compose([
        T.RectScale(height, width),
        T.ToTensor(),
        normalizer,
    ])

    train_loader = DataLoader(
        Preprocessor(train_set, root=root,
                     transform=train_transformer),
        batch_size=batchSize, num_workers=workers,
        sampler=RandomSampler(train_set),
       # shuffle=True,
        pin_memory=True, drop_last=True)

    val_loader = DataLoader(
        Preprocessor(val_list, root=root,
                     transform=test_transformer),
        batch_size=batchSize, num_workers=workers,
        shuffle=False, pin_memory=True)

    test_loader = DataLoader(
        Preprocessor(list(set(query_list) | set(gallery_list)),
                     root=root, transform=test_transformer),
        batch_size=batchSize, num_workers=workers,
        shuffle=False, pin_memory=True)

    return train_loader, val_loader, test_loader
def show_info(model,with_arch=True,with_grad=False):
    # TODO : show output info
    ### module info
    if with_arch:
        print('\n\n################### modules ################')
        for name,m in model.named_modules():
            print('{}:{}'.format(name,m))
        print('################### modules ################\n\n')
    #TODO: show grad status
    if with_grad:
        print('################### param diffs ################')
        for name,param in model.named_parameters():
            if 'classifier' not in name:
                if param.grad is not None:
                    print(name)
                    mean_data = torch.abs(param.data).mean()
                    mean_grad = torch.abs(param.grad).mean().data[0] + 1e-8
                    print('{}:size {},data:{},grad:{},data/grad:{}'.format(name,param.size(),mean_data,mean_grad,mean_data/mean_grad))
        print('################### param diffs ################')
    else:
        ### param info ###
        print('################### params ################')
        for name,param in model.named_parameters():
            if 'feat' not in name:
                print(name)
                mean_data = torch.abs(param.data).mean()
                print('{}:size {},abs_avg:{}'.format(name,
                                             param.size(),
                                             mean_data))
        print('################### params ################\n\n')
# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.01)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def load_params(new_model, pretrained_model):
    new_model_dict = new_model.state_dict()
    pretrained_checkpoint = load_checkpoint(pretrained_model)
    #for name, param in pretrained_checkpoint.items():
    for name, param in pretrained_checkpoint['state_dict'].items():
        print('pretrained_model params name and size: ', name, param.size())

        if name in new_model_dict :
            if isinstance(param, Parameter):
                # backwards compatibility for serialized parameters
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

def euclidean_distance(x1, x2, p=2, eps=1e-6):
    diff = torch.abs(x1 - x2)
    return torch.mean(torch.pow(diff + eps, p))

lr_init = opt.lr
###################### learninng rate #############################
def adjust_lr(opti, epoch):
    step_size = opt.step_size
    decay_step = opt.decay_step
    lr = opt.lr if epoch < step_size else \
        opt.lr * (0.1 ** ((epoch - step_size) // decay_step + 1))
    for g in opti.param_groups:
        g['lr'] = lr * g.get('lr_mult', 1)
    print('epoch: {}, lr: {}'.format(epoch, g['lr']))
##################### learninng rate #############################
###################### learninng rate #############################
def adjust_lr_un(opti, epoch):
    step_size = opt.step_size
    decay_step = opt.decay_step
    lr = opt.lr_un if epoch < step_size else \
        opt.lr_un * (0.1 ** ((epoch - step_size) // decay_step + 1))
    for g in opti.param_groups:
        g['lr'] = lr * g.get('lr_mult', 1)
    print('epoch: {}, lr: {}'.format(epoch, g['lr']))
##################### learninng rate #############################


def name_in_lst(name, lst):
    if lst is None:
        return False
    for ele in lst:
        if ele in name:
            return True
        return  False

def mult_lr(model,param_groups):
    param_and_info = defaultdict(list)
    name_and_info = defaultdict(list)
    for name,param  in  model.named_parameters():
        if name_in_lst(name, opt.spec_params):
            param_and_info[opt.spec_lr_mult].append(param)
            name_and_info[opt.spec_lr_mult].append(name)
        else:
            param_and_info[opt.base_lr_mult].append(param)
            name_and_info[opt.base_lr_mult].append(name)
        #else:
        #    param_and_info[1].append(param)
        #    name_and_info[1].append(name)
    for lr_mult in param_and_info.keys():
        param_groups.append({
                'params': param_and_info[lr_mult],
                'lr_mult': lr_mult
            })
        print('lr_mult: {},params: {}'.format(lr_mult, name_and_info[lr_mult]))
    return param_groups

input = torch.FloatTensor(opt.batchSize, 3, opt.height, opt.width)
label = torch.FloatTensor(opt.batchSize)
one = torch.FloatTensor([1])
mone = one * -1
if opt.cuda:
    input, label = input.cuda(), label.cuda()
    one = one.cuda()
    mone = mone.cuda()

def run():
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    cudnn.benchmark = True
    data_dir = opt.data_dir
# Redirect print to both console and log file
    #if not opt.evaluate:
    #    sys.stdout = Logger(osp.join(opt.logs_dir, 'log_l2_per.txt'))
    # Create data loaders
    def readlist(path):
        lines=[]
        with open(path, 'r') as f:
            data = f.readlines()

        #pdb.set_trace()
        for line in data:
            name, pid, cam = line.split()
            lines.append((name, int(pid), int(cam)))
        return lines

    # Load data list for wuzhen
    if osp.exists(osp.join(data_dir, 'train.txt')):
        train_list = readlist(osp.join(data_dir, 'train.txt'))
    else:
        print("The training list doesn't exist")

    if osp.exists(osp.join(data_dir, 'val.txt')):
        val_list = readlist(osp.join(data_dir, 'val.txt'))
    else:
        print("The validation list doesn't exist")

    if osp.exists(osp.join(data_dir, 'query.txt')):
        query_list = readlist(osp.join(data_dir, 'query.txt'))
    else:
        print("The query.txt doesn't exist")

    if osp.exists(osp.join(data_dir, 'gallery.txt')):
        gallery_list = readlist(osp.join(data_dir, 'gallery.txt'))
    else:
        print("The gallery.txt doesn't exist")

    if opt.height is None or opt.width is None:
        opt.height, opt.width = (144, 56) if opt.arch == 'inception' else \
                                  (256, 128)

    train_loader,val_loader, test_loader = \
        get_data(opt.split, data_dir, opt.height,
                 opt.width, opt.batchSize, opt.workers,
                 opt.combine_trainval, train_list, val_list, query_list, gallery_list)
    # Create model
     # ori 14514; clear 12654,  16645
    densenet = densenet121(num_classes = 20330,num_features = 256)
    start_epoch = best_top1 = 0
    if opt.resume:
        #checkpoint = load_checkpoint(opt.resume)
        #densenet.load_state_dict(checkpoint['state_dict'])
        densenet.load_state_dict(torch.load(opt.resume))
        start_epoch = opt.resume_epoch
        print("=> Finetune Start epoch {} "
         .format(start_epoch))
    if opt.pretrained_model:
         print('Start load params...')
         load_params(densenet,opt.pretrained_model)
    # Load from checkpoint
    #densenet = nn.DataParallel(densenet).cuda()
    metric = DistanceMetric(algorithm=opt.dist_metric)
    print('densenet')
    show_info(densenet,with_arch = True,with_grad = False)
    netG = netg()
    print('netG')
    show_info(netG,with_arch = True,with_grad = False)
    netG.apply(weights_init)
    if opt.netG != '':
        netG.load_state_dict(torch.load(opt.netG))
        #load_params(netG,opt.netG)
    if opt.cuda:
        netG = netG.cuda()
        densenet = densenet.cuda()
    perceptionloss=perception_loss(cuda = opt.cuda)
    l2loss=l2_loss(cuda = opt.cuda)
#    discriloss=discri_loss(cuda = opt.cuda,batchsize = opt.batchSize,height = \
#                           opt.height,width = opt.width,lr = opt.lr,step_size = \
#                           opt.step_size,decay_step = opt.decay_step )
    # Evaluator
    evaluator = Evaluator(densenet)
#    if opt.evaluate:
    metric.train(densenet, train_loader)
    print("Validation:")
    evaluator.evaluate(val_loader, val_list, val_list, metric)
    print("Test:")
    evaluator.evaluate(test_loader, query_list, gallery_list, metric)
    #    return
    # Criterion
#    criterion = nn.CrossEntropyLoss(ignore_index=-100).cuda()
    criterion = nn.CrossEntropyLoss().cuda()
    # Optimizer
    param_groups = []
    mult_lr(densenet,param_groups)
    optimizer = optim.SGD(param_groups, lr=opt.lr,
                                momentum=opt.momentum,
                                weight_decay=opt.weight_decay)
#    optimizer = optim.Adam(param_groups, lr=opt.lr, betas=(opt.beta1, 0.9))

    optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.9))

    # Start training
    for epoch in range(start_epoch, opt.epochs):
        adjust_lr(optimizer, epoch)
        adjust_lr(optimizerG, epoch)
        #discriloss.adjust_lr(epoch)
        losses = AverageMeter()
        precisions = AverageMeter()
        densenet.train()
        for i, data in enumerate(train_loader):
        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        # train with real
            real_cpu, _ , pids , _ = data
            if opt.cuda:
                real_cpu = real_cpu.cuda()
                targets = Variable(pids.cuda())
                input.resize_as_(real_cpu).copy_(real_cpu)
            inputv = Variable(input)
            outputs,output_dense,_ = densenet(inputv)
            fake = netG(output_dense)
            fake = fake * 3
            #discriloss(fake = fake, inputv = inputv, i = i)
        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
            if i % opt.CRITIC_ITERS == 0:
                netG.zero_grad()
                optimizer.zero_grad()
                #loss_discri = discriloss.gloss(fake = fake)
                loss_l2 = l2loss(fake = fake, inputv = inputv)
                loss_perception = perceptionloss(fake = fake, inputv = inputv)
                loss_classify = criterion(outputs, targets)
                prec, = accuracy(outputs.data, targets.data)
                prec = prec[0]
                losses.update(loss_classify.data[0], targets.size(0))
                precisions.update(prec, targets.size(0))
                loss = loss_classify + 0 * loss_l2 + 0 * loss_perception
#                loss = loss_discri
                loss.backward()
                optimizerG.step()
                optimizer.step()
            #print(precisions.val)
            #print(precisions.avg)
 #           print('[%d/%d][%d/%d] '%(epoch, opt.epochs, i, len(train_loader)))
#            print('[%d/%d][%d/%d] Loss_discri: %.4f '%(epoch, opt.epochs, i, \
#                  len(train_loader),loss_discri.data[0]))
            print('[%d/%d][%d/%d] Loss_l2: %.4f Loss_perception: %.4f '%(epoch, opt.epochs, i, \
                  len(train_loader),loss_l2.data[0],loss_perception.data[0]))
            print('Loss {}({})\t''Prec {}({})\t'.format(losses.val,losses.avg,precisions.val,precisions.avg))
            if i % 100 == 0:
                vutils.save_image(real_cpu,
                        '%s/real_samples.png' % opt.outf,
                        normalize=True)
                outputs,output_dense,_ = densenet(x = inputv)
                fake = netG(output_dense)
                fake = fake * 3
                vutils.save_image(fake.data,
                        '%s/fake_samples_epoch_%03d.png' % (opt.outf, epoch),
                         normalize=True)
        show_info(densenet,with_arch = False ,with_grad = True)
        show_info(netG,with_arch = False ,with_grad = True)
        if epoch % 5 == 0:
            torch.save(densenet.state_dict(), '%s/densenet_epoch_%d.pth' % (opt.outf, epoch))
            torch.save(netG.state_dict(), '%s/netG_epoch_%d.pth' % (opt.outf, epoch))
        if epoch < opt.start_save:
            continue
        top1 = evaluator.evaluate(val_loader, val_list, val_list)

        is_best = top1 > best_top1
        best_top1 = max(top1, best_top1)
        save_checkpoint({
            'state_dict': densenet.state_dict(),
            'epoch': epoch + 1,
            'best_top1': best_top1,
        }, is_best, fpath=osp.join(opt.logs_dir, 'checkpoint.pth.tar'))

        print('\n * Finished epoch {:3d}  top1: {:5.1%}  best: {:5.1%}{}\n'.
              format(epoch, top1, best_top1, ' *' if is_best else ''))
        if (epoch+1) % 5 == 0:
            print('Test model: \n')
            evaluator.evaluate(test_loader, query_list, gallery_list)
            model_name = 'epoch_'+ str(epoch) + '.pth.tar'
            torch.save({'state_dict':densenet.state_dict()},
                       osp.join(opt.logs_dir, model_name))
    # Final test
    print('Test with best model:')
    checkpoint = load_checkpoint(osp.join(opt.logs_dir, 'model_best.pth.tar'))
    densenet.load_state_dict(checkpoint['state_dict'])
    print('best epoch: ', checkpoint['epoch'])
    metric.train(densenet, train_loader)
    evaluator.evaluate(test_loader, query_list, gallery_list, metric)


if __name__ == '__main__':
    run()
