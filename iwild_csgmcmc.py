'''Train CIFAR10 with PyTorch.'''
from __future__ import print_function
import sys
sys.path.append('..')
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse

from models import *
from torch.autograd import Variable
from utils.dataloader import *
import numpy as np
import random
from utils.dataloader import get_wilds_dataloader
import pdb

parser = argparse.ArgumentParser(description='cSG-MCMC CIFAR10 Training')
parser.add_argument('--dir', type=str, default=None, required=True, help='path to save checkpoints (default: None)')
parser.add_argument('--data_path', type=str, default='data', metavar='PATH',
                    help='path to datasets location (default: None)')
parser.add_argument('--epochs', type=int, default=100,
                    help='number of epochs to train (default: 8)')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--alpha', type=int, default=1,
                    help='1: SGLD')
parser.add_argument('--device_id',type = int, help = 'device id to use')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed')
parser.add_argument('--temperature', type=float, default=1./129809,
                    help='temperature (default: 1/dataset_size)')
parser.add_argument('--topk', type=int, default=64)
parser.add_argument('--curricular', action='store_true')

args = parser.parse_args()
device_id = args.device_id
use_cuda = torch.cuda.is_available()
torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)
print("Arguments: ########################")
print('\n'.join(f'{k}={v}' for k, v in vars(args).items()))
print("###################################")
# Data
print('==> Preparing data..')
trainloader, testloader = get_wilds_dataloader("iwildcam", args) 

# Model
print('==> Building model..')
net = ResNet18(num_classes=182)
if use_cuda:
    net.cuda(device_id)
    cudnn.benchmark = True
    cudnn.deterministic = True

def noise_loss(lr,alpha):
    noise_loss = 0.0
    noise_std = (2/lr*alpha)**0.5
    for var in net.parameters():
        means = torch.zeros(var.size()).cuda(device_id)
        noise_loss += torch.sum(var * torch.normal(means, std = noise_std).cuda(device_id))
    return noise_loss

def adjust_learning_rate(optimizer, epoch, batch_idx):
    rcounter = epoch*num_batch+batch_idx
    cos_inner = np.pi * (rcounter % (T // M))
    cos_inner /= T // M
    cos_out = np.cos(cos_inner) + 1
    lr = 0.5*cos_out*lr_0

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (inputs, targets, metadata) in enumerate(trainloader):
        if use_cuda:
            inputs, targets = inputs.cuda(device_id), targets.cuda(device_id)

        optimizer.zero_grad()
        lr = adjust_learning_rate(optimizer, epoch,batch_idx)
        outputs = net(inputs)
        if (epoch%25)+1 > 20:
            loss_noise = noise_loss(lr,args.alpha)*(args.temperature/datasize)**.5
            loss = criterion(outputs, targets)
            if args.curricular and len(loss) > args.topk:
                loss = torch.topk(loss, args.topk, dim=0).values.mean()
            loss += loss_noise
        #elif (epoch % 50) + 1 < 3:
        #    loss = criterion(outputs, targets)
            #if args.curricular and len(loss) > args.topk:
            #    loss = torch.topk(loss, args.topk, dim=0, largest=False).values.mean()
        else:
            loss = criterion(outputs, targets).mean()
        loss.mean().backward()
        optimizer.step()

        train_loss += loss.mean().data.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()
        if batch_idx%100==0:
            print('Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (train_loss/(batch_idx+1), 100.*correct.item()/total, correct, total))


def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets, metadata) in enumerate(testloader):
            if use_cuda:
                inputs, targets = inputs.cuda(device_id), targets.cuda(device_id)
            outputs = net(inputs)
            loss = criterion(outputs, targets).mean()

            test_loss += loss.data.item()
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()

            if batch_idx%100==0:
                print('Test Loss: %.3f | Test Acc: %.3f%% (%d/%d)'
                    % (test_loss/(batch_idx+1), 100.*correct.item()/total, correct, total))

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
    test_loss/len(testloader), correct, total,
    100. * correct.item() / total))

datasize = 129809
num_batch = datasize/args.batch_size+1
lr_0 = 0.5 # initial lr
M = 4 # number of cycles
T = args.epochs*num_batch # total number of iterations
print(f"Num batch: [{num_batch}], cycles: [{M}], iterations: [{T}]")
criterion = nn.CrossEntropyLoss(reduction='none')
optimizer = optim.SGD(net.parameters(), lr=lr_0, momentum=1-args.alpha, weight_decay=5e-4)
mt = 0

for epoch in range(args.epochs):
    train(epoch)
    test(epoch)
    if (epoch%25)+1>22: # save 3 models per cycle
        print('save!')
        net.cpu()
        torch.save(net.state_dict(),args.dir + '/cifar_model_%i.pt'%(mt))
        mt += 1
        net.cuda(device_id)

