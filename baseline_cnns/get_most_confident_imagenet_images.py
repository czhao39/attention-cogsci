"""
Evaluates CIFAR-100 models on ImageNet and gets the most confidently classified ImageNet image in each of a given set of CIFAR-100 classes.
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

parser.add_argument('-o', '--output', type=str, default='cifar_class_to_most_conf_imagenet.pkl', help='path to output file')
# Datasets
parser.add_argument('-d', '--data', type=str, required=True, help='path to dataset',)
parser.add_argument('-j', '--workers', default=1, type=int, metavar='N',
                    help='number of data loading workers (default: 1)')
parser.add_argument('-m', '--matches', type=str, required=True, help='path to pickle file containing dictionary that maps desired ImageNet classes to CIFAR-100 classes')
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
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.Resize(32),  # For input into CIFAR-100 models
        transforms.ToTensor(),
        normalize,
    ])

    val_dataset = datasets.ImageNet(root=args.data, split="val", transform=val_transform)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
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

    # Load map of ImageNet to CIFAR-100 classes
    with open(args.matches, "rb") as infile:
        imagenet_to_cifar = pickle.load(infile)
    cifar_to_imagenet = {}
    for imagenet_class, cifar_class in imagenet_to_cifar.items():
        if cifar_class in cifar_to_imagenet:
            cifar_to_imagenet[cifar_class].add(imagenet_class)
        else:
            cifar_to_imagenet[cifar_class] = {imagenet_class}
    print(f"{len(imagenet_to_cifar)} ImageNet classes mapped to {len(cifar_to_imagenet)} CIFAR-100 classes")

    # Maps CIFAR-100 classes to confidence and path of most confidently correctly predicted
    cifar_class_to_most_conf = {}

    batch_time = AverageMeter()
    accuracy = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        bar = Bar('Processing', max=len(val_loader))
        for batch_idx, (inputs, targets) in enumerate(val_loader):
            target = targets[0]
            target_imagenet_names = val_dataset.classes[target]
            for n in target_imagenet_names:
                if n in imagenet_to_cifar:
                    target_cifar_name = imagenet_to_cifar[n]
                    break
            else:
                bar.next()
                continue

            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)

            # compute output
            output = nn.functional.softmax(model(inputs)).cpu().numpy()[0]

            pred = np.argmax(output)
            pred_cifar_name = cifar_classes[pred]

            # record CIFAR accuracy
            acc = (pred_cifar_name == target_cifar_name)
            accuracy.update(acc, 1)

            # update most confident
            if acc and (pred_cifar_name not in cifar_class_to_most_conf or output[pred] > cifar_class_to_most_conf[pred_cifar_name][0]):
                cifar_class_to_most_conf[pred_cifar_name] = (output[pred], val_dataset.imgs[batch_idx][0])

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # plot progress
            bar.suffix  = '({batch}/{size}) Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Accuracy: {acc:.4f}'.format(
                        batch=batch_idx + 1,
                        size=len(val_loader),
                        bt=batch_time.avg,
                        total=bar.elapsed_td,
                        eta=bar.eta_td,
                        acc=accuracy.avg,
                        )
            bar.next()
        bar.finish()

        pprint(cifar_class_to_most_conf)
        print(len(cifar_class_to_most_conf))
        with open(args.output, "wb") as outfile:
            pickle.dump(cifar_class_to_most_conf, outfile)

if __name__ == '__main__':
    main()
