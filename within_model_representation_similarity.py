import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import scipy.cluster.hierarchy as sch
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


# Parse arguments
parser = argparse.ArgumentParser(description="Visualize similarities between internal representations")

parser.add_argument("-r", "--representations", type=str, required=True, help="path to numpy file containing image internal representations, one per row")
parser.add_argument("-i", "--imagedir", type=str, required=True, help="path to root directory of images")

args = parser.parse_args()


def cluster_corr(corr_array):
    # Taken from https://wil.yegelwel.com/cluster-correlation-matrix/
    """
    Rearranges the correlation matrix, corr_array, so that groups of highly 
    correlated variables are next to eachother 

    Parameters
    ----------
    corr_array : pandas.DataFrame or numpy.ndarray
        a NxN correlation matrix 

    Returns
    -------
    pandas.DataFrame or numpy.ndarray
        a NxN correlation matrix with the columns and rows rearranged
    """
    pairwise_distances = sch.distance.pdist(corr_array, metric="correlation")
    linkage = sch.linkage(pairwise_distances, method='complete')
    cluster_distance_threshold = pairwise_distances.max()/2
    idx_to_cluster_array = sch.fcluster(linkage, cluster_distance_threshold,
                                        criterion='distance')
    idx = np.argsort(idx_to_cluster_array)

    if isinstance(corr_array, pd.DataFrame):
        return corr_array.iloc[idx, :].T.iloc[idx, :]
    return corr_array[idx, :][:, idx]


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
    #corr_df = cluster_corr(corr_df)
    sns.set(style="whitegrid", font_scale=0.8)
    plt.figure(figsize=(8, 8))
    plt.tick_params(axis="both", labelsize=4, pad=1, length=0)
    sns.heatmap(corr_df, square=True, cmap="bwr", cbar=True, vmin=-1, vmax=1)
    unique_classes = list(np.unique(classes))
    cmap = plt.cm.tab10(np.linspace(0, 1, len(unique_classes)))
    for i, xy_tick in enumerate(zip(plt.gca().xaxis.get_ticklabels(), plt.gca().yaxis.get_ticklabels())):
        the_class = corr_df.index[i][:corr_df.index[i].index("/")]
        color = cmap[unique_classes.index(the_class)]
        xy_tick[0].set_color(color)
        xy_tick[1].set_color(color)
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
