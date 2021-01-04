import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from model1 import AttnVGG_before
from model2 import AttnVGG_after


#use_cuda = torch.cuda.is_available()
use_cuda = False
device = torch.device("cuda" if use_cuda else "cpu")

parser = argparse.ArgumentParser(description="LearnToPayAttn-CIFAR100")

parser.add_argument("-o1", "--confidences_out", type=str, default="confidences.txt", help="path to output file")
parser.add_argument("-o2", "--representations_out", type=str, default="representations.npy", help="path to output file")
parser.add_argument("--image_dir", "-i", required=True, help="path to images")
parser.add_argument("--model", "-m", required=True, help="path to model")
parser.add_argument("--attn_mode", type=str, default="before", help="insert attention modules before OR after maxpooling layers")
parser.add_argument("--normalize_attn", action="store_true", help="if True, attention map is normalized by softmax; otherwise use sigmoid")
parser.add_argument('-c', '--cifar100-classes', type=str, required=True, help='path to txt file containing list of CIFAR-100 classes')

opt = parser.parse_args()


def main():
    im_size = 32
    normalize = transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    transform = transforms.Compose([
        transforms.Resize(im_size),
        transforms.ToTensor(),
        normalize,
    ])

    dataset = datasets.ImageFolder(root=opt.image_dir, transform=transform)
    loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=1, shuffle=False,
            num_workers=1, pin_memory=True)
    print('done')

    ## load network
    print('\nloading the network ...\n')
    # (linear attn) insert attention befroe or after maxpooling?
    # (grid attn only supports "before" mode)
    if opt.attn_mode == 'before':
        print('\npay attention before maxpooling layers...\n')
        net = AttnVGG_before(im_size=im_size, num_classes=100,
                attention=True, normalize_attn=opt.normalize_attn, init='xavierUniform')
    elif opt.attn_mode == 'after':
        print('\npay attention after maxpooling layers...\n')
        net = AttnVGG_after(im_size=im_size, num_classes=100,
                attention=True, normalize_attn=opt.normalize_attn, init='xavierUniform')
    else:
        raise NotImplementedError("Invalid attention mode!")
    print('done')

    ## load model
    print('\nloading the model ...\n')
    state_dict = torch.load(opt.model, map_location=str(device))
    # Remove 'module.' prefix
    state_dict = {k[7:]: v for k, v in state_dict.items()}
    net.load_state_dict(state_dict)
    net = net.to(device)
    net.eval()
    print('done')

    model = net

    # base factor
    if opt.attn_mode == 'before':
        min_up_factor = 1
    else:
        min_up_factor = 2

    # Add hook to get internal representation
    global internal_repr
    def update_internal_repr(module, inpt, output):
        global internal_repr
        #internal_repr = output.flatten()
        internal_repr = inpt[0].flatten()
        print(internal_repr[100])
        print(inpt[0].flatten()[10])
    model.classify.register_forward_hook(update_internal_repr)
    #model.dense.register_forward_hook(update_internal_repr)
    #model.conv_block6.register_forward_hook(update_internal_repr)
    print(model)

    # Load CIFAR-100 classes
    with open(opt.cifar100_classes, "r") as infile:
        cifar_classes = infile.read().split()
    assert(len(cifar_classes) == 100)

    results = []

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(loader):
            output = F.softmax(model(inputs)[0]).cpu().numpy()[0]

            top5_inds = np.argpartition(output, -5)[-5:]
            top5 = [(cifar_classes[ind], output[ind]) for ind in top5_inds]
            top5.sort(key=lambda x: x[1], reverse=True)

            rep = internal_repr.cpu().numpy()

            results.append((dataset.imgs[batch_idx][0], *(x for tup in top5 for x in tup), rep))

    results.sort()
    with open(opt.confidences_out, "w") as outfile:
        outfile.write("\n".join("\t".join(map(str, res[:-1])) for res in results))
    print("Wrote confidences to", opt.confidences_out)

    reprs = np.array([res[-1] for res in results])
    np.save(opt.representations_out, reprs)
    print("Wrote internal representations to", opt.representations_out)


if __name__ == "__main__":
    main()
