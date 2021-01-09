import argparse
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import models.cifar as models

import numpy as np

from utils import Bar


model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

#use_cuda = torch.cuda.is_available()
use_cuda = False
device = torch.device("cuda" if use_cuda else "cpu")

parser = argparse.ArgumentParser(description='ABN-CIFAR100')
parser.add_argument('-oc', '--confidences_out', type=str, default='confidences.txt', help='path to output file')
parser.add_argument('-or', '--representations_out', type=str, default='representations.npy', help='path to output file')
parser.add_argument('-os', '--softmaxes_out', type=str, default='softmaxes.npy', help='path to output file')
# Datasets
parser.add_argument('-d', '--data', type=str, required=True, help='path to dataset',)
parser.add_argument('-j', '--workers', default=1, type=int, metavar='N',
                    help='number of data loading workers (default: 1)')
parser.add_argument('-c', '--cifar100-classes', type=str, required=True, help='path to txt file containing list of CIFAR-100 classes')
parser.add_argument('--model', '-m', default='', type=str, metavar='PATH',
                    help='path to model (default: none)')
# Architecture
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet)')
parser.add_argument('--depth', type=int, default=110, help='Model depth.')
parser.add_argument('--cardinality', type=int, default=8, help='Model cardinality (group).')
parser.add_argument('--widen-factor', type=int, default=4, help='Widen factor. 4 -> 64, 8 -> 128, ...')
parser.add_argument('--growthRate', type=int, default=12, help='Growth rate for DenseNet.')
parser.add_argument('--compressionRate', type=int, default=2, help='Compression Rate (theta) for DenseNet.')
parser.add_argument('--drop', '--dropout', default=0, type=float,
        metavar='Dropout', help='Dropout ratio')

args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}


def main():
    im_size = 32
    num_classes = 100
    normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    transform = transforms.Compose([
        transforms.Resize(im_size),
        transforms.ToTensor(),
        normalize,
    ])

    dataset = datasets.ImageFolder(root=args.data, transform=transform)
    loader = torch.utils.data.DataLoader(dataset,
                                         batch_size=1, shuffle=False,
                                         num_workers=args.workers, pin_memory=True)

    # Model
    print("==> creating model '{}'".format(args.arch))
    if args.arch.startswith('resnext'):
        model = models.__dict__[args.arch](
                    cardinality=args.cardinality,
                    num_classes=num_classes,
                    depth=args.depth,
                    widen_factor=args.widen_factor,
                    dropRate=args.drop,
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
                )
    else:
        model = models.__dict__[args.arch](num_classes=num_classes)

    model = torch.nn.DataParallel(model)
    if use_cuda:
        model = model.cuda()
    print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))

    # Load model
    title = 'cifar-100-' + args.arch
    if args.model:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isfile(args.model), 'Error: no checkpoint directory found!'
        checkpoint = torch.load(args.model)
        model.load_state_dict(checkpoint['state_dict'])

    # Add hook to get internal representation
    global internal_repr
    def update_internal_repr(module, inpt, output):
        global internal_repr
        internal_repr = inpt[0].flatten()
    model.module.fc.register_forward_hook(update_internal_repr)
    print(model)

    # Load CIFAR-100 classes
    with open(args.cifar100_classes, "r") as infile:
        cifar_classes = infile.read().split()
    assert(len(cifar_classes) == 100)

    # switch to evaluate mode
    model.eval()

    results = []

    bar = Bar('Processing', max=len(loader))
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(loader):
            output = nn.functional.softmax(model(inputs)[1]).cpu().numpy()[0]

            top5_inds = np.argpartition(output, -5)[-5:]
            top5 = [(cifar_classes[ind], output[ind]) for ind in top5_inds]
            top5.sort(key=lambda x: x[1], reverse=True)

            rep = internal_repr.cpu().numpy()

            results.append((dataset.imgs[batch_idx][0], *(x for tup in top5 for x in tup), rep, output))

            bar.next()
        bar.finish()

    results.sort()
    with open(args.confidences_out, "w") as outfile:
        outfile.write("\n".join("\t".join(map(str, res[:-2])) for res in results))
    print("Wrote confidences to", args.confidences_out)

    reprs = np.array([res[-2] for res in results])
    np.save(args.representations_out, reprs)
    print("Wrote internal representations to", args.representations_out)

    softmaxes = np.array([res[-1] for res in results])
    np.save(args.softmaxes_out, softmaxes)
    print("Wrote softmaxes to", args.softmaxes_out)


if __name__ == '__main__':
    main()
