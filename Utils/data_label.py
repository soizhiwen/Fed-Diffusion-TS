import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from collections import Counter
import numpy.random as npr
from torch.utils.data import Subset
from typing import List, cast
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

from iid_metrics import KS_test, compute_mutual_info

def data_label(
    file_path,
    window_size = 24,
    n_clusters = 5,
    step_size = 1
    ):

    data = pd.read_csv(file_path)
    subsets = [data.iloc[i:i+window_size] for i in range(0, len(data) - window_size + 1, step_size)]
    print(len(subsets))

    flattened_subsets = [np.array(subset).flatten() for subset in subsets]
    kmeans_models = KMeans(n_clusters=n_clusters).fit(flattened_subsets)
    labels = kmeans_models.labels_
    unique_labels = np.unique(kmeans_models.labels_)
    print(unique_labels)

    label_counts = Counter(kmeans_models.labels_)

    for label, count in label_counts.items():
        print(f"Label {label}: {count} times")

    labeled_dataset = [[subsets[i], labels[i]] for i in range(len(subsets))]
    # labeled_dataset[ [[24x6], label] ]

    return labeled_dataset


def split(dataset, nr_clients: int, iid: bool, seed: int) -> List[Subset]:
    rng = npr.default_rng(seed)

    if iid:
        splits = np.array_split(rng.permutation(len(dataset)), nr_clients)
    else:
        sorted_indices = np.argsort(np.array([target for _data, target in dataset]))
        shards = np.array_split(sorted_indices, 2 * nr_clients)
        shuffled_shard_indices = rng.permutation(len(shards))
        splits = [
            np.concatenate([shards[i] for i in inds], dtype=np.int64)
            for inds in shuffled_shard_indices.reshape(-1, 2)
        ]

    return [Subset(dataset, split) for split in cast(List[List[int]], splits)]

labeled_dataset = data_label(file_path = 'Data/datasets/stock_data.csv')
sample_split = split(labeled_dataset, 100, True, 42)
statistic, p_value = KS_test(sample_split)
average_mi, total_mi_inside = compute_mutual_info(sample_split)
print(f"Average Mutual Information between all subsets: {average_mi}")
print(f"Average Mutual Information inside each subset: {total_mi_inside}")