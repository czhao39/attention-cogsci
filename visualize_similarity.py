import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA


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

    # Compute and display covariance matrix
    cov = np.cov(rep)
    cov_df = pd.DataFrame(cov, index=paths, columns=paths)
    sns.set(style="whitegrid", font_scale=0.2, rc={"xtick.major.pad": -5, "ytick.major.pad": -5})
    plt.figure(figsize=(8, 8))
    sns.heatmap(cov_df, square=True, cbar=False)
    plt.tight_layout()
    plt.savefig("covariance.png", dpi=400, bbox_inches="tight")
    plt.show()

    # Compute and display projections onto first two principal components
    pca = PCA(n_components=2)
    projected = pca.fit_transform(rep)
    sns.set(style="whitegrid", font_scale=0.7)
    plt.figure(figsize=(6, 6))
    sns.scatterplot(x=projected[:, 0], y=projected[:, 1], hue=classes)
    for i, label in enumerate(paths):
        plt.annotate(label, projected[i], fontsize=4)
    plt.tight_layout()
    plt.savefig("pca.png", dpi=300, bbox_inches="tight")
    plt.show()



if __name__ == "__main__":
    main()
