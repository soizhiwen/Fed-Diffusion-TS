import pandas as pd
import numpy as np
from sklearn.cluster import KMeans


def label_data(path, window_size=24, n_clusters=5, step_size=1):
    data = pd.read_csv(path)
    subsets = [
        data.iloc[i : i + window_size].values
        for i in range(0, len(data) - window_size + 1, step_size)
    ]
    flattened_subsets = [np.array(subset).flatten() for subset in subsets]
    kmeans_models = KMeans(n_clusters=n_clusters).fit(flattened_subsets)
    labels = kmeans_models.labels_
    labeled_dataset = [[subsets[i], labels[i]] for i in range(len(subsets))]
    return labeled_dataset


if __name__ == "__main__":
    dataset = label_data(
        path="../../Data/datasets/stock_data.csv",
        window_size=24,
        n_clusters=5,
        step_size=1,
    )
    np.save("../../Data/datasets/labeled_stock_data.npy", dataset)
