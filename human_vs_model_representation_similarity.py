import argparse
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


sns.set_style("whitegrid")


# Parse arguments
parser = argparse.ArgumentParser(description="Visualize similarities between human and model representations")

parser.add_argument("-i", "--imagedir", type=str, required=True, help="path to root directory of images")
parser.add_argument("-s", "--softmaxes", type=str, required=True, help="path to numpy file containing softmax activations, one per row")
#parser.add_argument("-r", "--representations", type=str, required=True, help="path to numpy file containing image internal representations, one per row")
parser.add_argument("-c", "--correlations", type=str, required=True, help="path to pickle file containing dictionary from image names to attention map correlations with human attention maps")
parser.add_argument("--cifar100-classes", type=str, required=True, help="path to txt file containing ordered list of CIFAR-100 classes")
parser.add_argument("--cifar100-superclass-to-classes", type=str, required=True, help="path to pickle file containing dictionary from CIFAR-100 superclasses to classes")

args = parser.parse_args()


def main():
    # Get image paths
    paths = []
    superclasses = []
    names = []
    for dirpath, dirnames, filenames in os.walk(args.imagedir):
        for f in filenames:
            if f.endswith(".jpg"):
                superclasses.append(os.path.basename(dirpath))
                names.append(f)
                paths.append(os.path.join(superclasses[-1], names[-1]))
    zipped = list(zip(paths, superclasses, names))
    zipped.sort()
    paths, superclasses, names = zip(*zipped)
    print("Loaded image paths")

    # Load CIFAR-100 metadata
    with open(args.cifar100_classes, "r") as infile:
        index_to_class = infile.read().split()
    class_to_index = {c: i for i, c in enumerate(index_to_class)}
    with open(args.cifar100_superclass_to_classes, "rb") as infile:
        superclass_to_classes = pickle.load(infile)
    # Combine vehicles_1 and vehicles_2 superclasses
    superclass_to_classes["vehicles"] = superclass_to_classes["vehicles_1"] | superclass_to_classes["vehicles_2"]
    del superclass_to_classes["vehicles_1"]
    del superclass_to_classes["vehicles_2"]
    superclass_to_class_inds = {superclass: [class_to_index[c] for c in classes] for superclass, classes in superclass_to_classes.items()}
    print("Loaded CIFAR-100 metadata")

    # Load internal representations
    #reps = []
    #for reppath in args.representations:
    #    reps.append(np.load(reppath))
    #    assert(reps[-1].shape[0] == len(paths))
    #    print("Loaded representations from", reppath)

    # Load softmaxes and correlations
    softmaxes = np.load(args.softmaxes)
    with open(args.correlations, "rb") as infile:
        corrs = pickle.load(infile)
    corrs = np.array([corrs[n] for n in names])
    assert (len(paths) == len(softmaxes) == len(corrs))
    print("Loaded softmaxes and correlations")

    # Compute amount of confidence in correct superclass
    superclass_confs = []
    num_correct = 0
    for i in range(len(softmaxes)):
        correct_class_inds = superclass_to_class_inds[superclasses[i]]
        num_correct += softmaxes[i].argmax() in correct_class_inds
        superclass_confs.append(softmaxes[i, correct_class_inds].sum())
        #max_conf = softmaxes[i, correct_class_inds].max()
        #superclass_confs[-1] = (superclass_confs[-1] - max_conf) / (1.0 - max_conf)
    print(f"Model superclass accuracy: {num_correct} / {len(softmaxes)} = {num_correct / len(softmaxes)}")

    # TODO: for y-axis, try using number of nearest 10 neighbors in correct superclass divided by total number in correct superclass?

    # Generate figure
    #fig, axs = plt.subplots(1, corrs.shape[1], figsize=(4 * corrs.shape[1], 5))
    #plt.figure(figsize=(5, 5))
    hues = []
    human_attn_types = []
    for typ in ["free-viewing", "saliency-viewing", "object search", "explicit judgment"]:
        human_attn_types.extend([typ] * corrs.shape[0])
    df = pd.DataFrame({"Corr b/w Model and Human Attn Maps": corrs.ravel(),
                       "Model Conf in Correct Superclass": superclass_confs * corrs.shape[1],
                       "Human Attention Map": human_attn_types})
    #sns.scatterplot(data=df, x="Corr b/w Model and Human Attn Maps", y="Model Conf in Correct Superclass", hue="Human Attention Map")
    g = sns.FacetGrid(df, col="Human Attention Map")
    g.map(sns.scatterplot, "Corr b/w Model and Human Attn Maps", "Model Conf in Correct Superclass")
    plt.tight_layout()
    plt.savefig("representationsim_vs_attentionsim.png", dpi=200, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main()
