import pickle
from pprint import pprint
import string


def process_classname(name):
    trans = str.maketrans(string.punctuation, " "*len(string.punctuation))
    return name.lower().translate(trans)
    #return ''.join(c for c in name.lower() if c.isalnum())

with open("cifar_100_classes.txt", "r") as infile:
    cifar_classes = [(process_classname(name), name) for name in infile.read().split()]

with open("imagenet1000_clsid_to_human.pkl", "rb") as infile:
    imagenet_idx_to_labels = pickle.load(infile)
imagenet_classes = [[(process_classname(label), label) for label in labels.split(", ")] for labels in imagenet_idx_to_labels.values()]

#matches = set(cifar_classes) & set(imagenet_classes)
matches = []
for in_cls in imagenet_classes:
    matched = False
    for processed_label, label in in_cls:
        for processed_cifar_cls, cifar_cls in cifar_classes:
            if processed_label.endswith(processed_cifar_cls) and (len(processed_label) == len(processed_cifar_cls) or processed_label[-len(processed_cifar_cls)-1] == " "):
                matches.append((label, cifar_cls))
                matched = True
                break
        if matched:
            break
imagenet_matches = {m[0] for m in matches}
cifar_matches = {m[1] for m in matches}

print(f"{len(imagenet_matches)} ImageNet classes matched to {len(cifar_matches)} CIFAR-100 classes")

imagenet_to_cifar = dict(matches)
assert len(imagenet_to_cifar) == len(imagenet_matches)
with open("matched_imagenet_to_cifar.pkl", "wb") as outfile:
    pickle.dump(imagenet_to_cifar, outfile)

cifar_to_imagenet = {}
for m in matches:
    if m[1] not in cifar_to_imagenet:
        cifar_to_imagenet[m[1]] = {m[0]}
    else:
        cifar_to_imagenet[m[1]].add(m[0])
pprint(cifar_to_imagenet)

# with open("cifar-100-superclasses-to-classes.pkl", "rb") as infile:
#     coarse_to_fine = pickle.load(infile)
#
# fine_to_coarse = {}
# for coarse, fines in coarse_to_fine.items():
#     for fine in fines:
#         fine_to_coarse[fine] = coarse
#
# matched_coarses = {fine_to_coarse[fine] for fine in cifar_matches}
# pprint(matched_coarses)
# print(len(matched_coarses))
