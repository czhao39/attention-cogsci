"""
Evaluates CIFAR-100 models on a given directory of images.
"""

from __future__ import print_function

import argparse
import os
import pickle
from pprint import pprint
import time
import random

import numpy as np

import torch
import torch.nn as nn
import torch.nn.parallel
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import models.cifar as models

from utils import Bar, Logger, AverageMeter, accuracy, mkdir_p, savefig


model_names = sorted(name for name in models.__dict__
        if name.islower() and not name.startswith("__")
        and callable(models.__dict__[name]))

# Parse arguments
parser = argparse.ArgumentParser(description='Evaluation of CIFAR-100 models on ImageNet')

parser.add_argument('-o', '--output', type=str, default='confidences.txt', help='path to output file')
# Datasets
parser.add_argument('-d', '--data', type=str, required=True, help='path to dataset',)
parser.add_argument('-j', '--workers', default=1, type=int, metavar='N',
                    help='number of data loading workers (default: 1)')
parser.add_argument('-c', '--cifar100-classes', type=str, required=True, help='path to txt file containing list of CIFAR-100 classes')
# Checkpoints
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
# Architecture
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet)')
parser.add_argument('--depth', type=int, default=110, help='Model depth.')
parser.add_argument('--block-name', type=str, default='BasicBlock',
        help='the building block for Resnet and Preresnet: BasicBlock, Bottleneck (default: Basicblock for cifar10/cifar100)')
parser.add_argument('--cardinality', type=int, default=8, help='Model cardinality (group).')
parser.add_argument('--widen-factor', type=int, default=4, help='Widen factor. 4 -> 64, 8 -> 128, ...')
parser.add_argument('--growthRate', type=int, default=12, help='Growth rate for DenseNet.')
parser.add_argument('--compressionRate', type=int, default=2, help='Compression Rate (theta) for DenseNet.')
# Miscs
parser.add_argument('--manualSeed', type=int, help='manual seed')

args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}

# Use CUDA
#os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
#use_cuda = torch.cuda.is_available()
use_cuda = False
device = torch.device("cuda" if use_cuda else "cpu")

# Random seed
if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)
if use_cuda:
    torch.cuda.manual_seed_all(args.manualSeed)

def main():
    # Data loading code
    normalize = transforms.Normalize((0.4914, 0.4822, 0.4465),
                                     (0.2023, 0.1994, 0.2010))  # CIFAR-100 normalization
    transform = transforms.Compose([
        #transforms.Resize(256),
        #transforms.CenterCrop(224),
        transforms.Resize(32),  # For input into CIFAR-100 models
        transforms.ToTensor(),
        normalize,
    ])

    dataset = datasets.ImageFolder(root=args.data, transform=transform)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    # Model
    print("==> creating model '{}'".format(args.arch))
    num_classes = 100
    if args.arch.startswith('resnext'):
        model = models.__dict__[args.arch](
                cardinality=args.cardinality,
                num_classes=num_classes,
                depth=args.depth,
                widen_factor=args.widen_factor,
                )
    elif args.arch.startswith('densenet'):
        model = models.__dict__[args.arch](
                num_classes=num_classes,
                depth=args.depth,
                growthRate=args.growthRate,
                compressionRate=args.compressionRate,
                )
    elif args.arch.startswith('wrn'):
        model = models.__dict__[args.arch](
                num_classes=num_classes,
                depth=args.depth,
                widen_factor=args.widen_factor,
                )
    elif args.arch.endswith('resnet'):
        model = models.__dict__[args.arch](
                num_classes=num_classes,
                depth=args.depth,
                block_name=args.block_name,
                )
    else:
        model = models.__dict__[args.arch](num_classes=num_classes)

    model = torch.nn.DataParallel(model)
    if use_cuda:
        model = model.cuda()
    print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))

    # Resume
    title = args.arch
    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isfile(args.resume), 'Error: no checkpoint directory found!'
        args.checkpoint = os.path.dirname(args.resume)
        checkpoint = torch.load(args.resume)
        best_acc = checkpoint['best_acc']
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])

    # Load CIFAR-100 classes
    with open(args.cifar100_classes, "r") as infile:
        cifar_classes = infile.read().split()
    assert(len(cifar_classes) == 100)

    batch_time = AverageMeter()

    results = []

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        bar = Bar('Processing', max=len(loader))
        for batch_idx, (inputs, targets) in enumerate(loader):
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)

            # compute output
            output = nn.functional.softmax(model(inputs)).cpu().numpy()[0]

            top5_inds = np.argpartition(output, -5)[-5:]
            top5 = [(cifar_classes[ind], output[ind]) for ind in top5_inds]
            top5.sort(key=lambda x: x[1], reverse=True)

            results.append((dataset.imgs[batch_idx][0], *(x for tup in top5 for x in tup)))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # plot progress
            bar.suffix  = '({batch}/{size}) Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:}'.format(
                        batch=batch_idx + 1,
                        size=len(loader),
                        bt=batch_time.avg,
                        total=bar.elapsed_td,
                        eta=bar.eta_td,
                        )
            bar.next()
        bar.finish()

    results.sort(reverse=True)
    with open(args.output, "w") as outfile:
        outfile.write("\n".join("\t".join(map(str, res)) for res in results))


if __name__ == '__main__':
    main()
