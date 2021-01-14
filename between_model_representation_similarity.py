import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


# Parse arguments
parser = argparse.ArgumentParser(description="Visualize similarities between internal representations across different models")

parser.add_argument("-r", "--representations", type=str, action="append", required=True, help="path to numpy file containing image internal representations, one per row; use multiple times to compare models")
parser.add_argument("-i", "--imagedir", type=str, required=True, help="path to root directory of images")

args = parser.parse_args()


def main():
    # Get image paths
    paths = []
    classes = []
    names = []
    for dirpath, dirnames, filenames in os.walk(args.imagedir):
        for f in filenames:
            if f.endswith(".jpg"):
                classes.append(os.path.basename(dirpath))
                names.append(f)
                paths.append(os.path.join(classes[-1], names[-1]))
    zipped = list(zip(paths, classes, names))
    zipped.sort()
    paths, classes, names = zip(*zipped)
    print("Loaded image paths")

    # Load internal representations
    reps = []
    for reppath in args.representations:
        reps.append(np.load(reppath))
        assert(reps[-1].shape[0] == len(paths))
        print("Loaded representations from", reppath)

    # Compute correlations for each model
    inds = np.triu_indices(len(paths))
    corrs = [np.corrcoef(rep)[inds] for rep in reps]
    print("Computed image representation correlations for each model")

    # Compute correlations across models
    corr = np.corrcoef(corrs)
    corr_df = pd.DataFrame(corr, index=args.representations, columns=args.representations)
    #sns.set(style="whitegrid", font_scale=0.2, rc={"xtick.major.pad": -5, "ytick.major.pad": -5})
    sns.set_style("whitegrid")
    plt.figure(figsize=(6, 6))
    sns.heatmap(corr_df, annot=True, square=True, cbar=True, vmin=-1, vmax=1)
    plt.tight_layout()
    plt.savefig("correlation.png", dpi=200, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main()
