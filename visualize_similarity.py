import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


# Parse arguments
parser = argparse.ArgumentParser(description="Visualize similarities between internal representations")

parser.add_argument("-r", "--representations", type=str, required=True, help="path to numpy file containing image internal representations, one per row")
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
    rep = np.load(args.representations)
    assert(rep.shape[0] == len(paths))
    print("Loaded representations")

    # Compute and display correlation matrix
    corr = np.corrcoef(rep)
    corr_df = pd.DataFrame(corr, index=paths, columns=paths)
    sns.set(style="whitegrid", font_scale=0.2, rc={"xtick.major.pad": -5, "ytick.major.pad": -5})
    plt.figure(figsize=(8, 8))
    sns.heatmap(corr_df, square=True, cbar=False)
    plt.tight_layout()
    plt.savefig("correlation.png", dpi=400, bbox_inches="tight")
    plt.show(block=False)

    # Compute and display 2D embedding
    projected = PCA(n_components=50).fit_transform(rep)
    projected = TSNE(n_components=2, init="pca", learning_rate=20, verbose=2).fit_transform(projected)
    sns.set(style="whitegrid", font_scale=0.7)
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=projected[:, 0], y=projected[:, 1], hue=classes)
    for i, label in enumerate(paths):
        plt.annotate(label, projected[i], fontsize=4)
    plt.tight_layout()
    plt.savefig("embedding.png", dpi=300, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main()
