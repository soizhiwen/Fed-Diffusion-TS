import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from collections import Counter
import numpy.random as npr
from torch.utils.data import Subset
from typing import List, cast
import warnings
warnings.filterwarnings("ignore")

from iid_metrics import KS_test, compute_mutual_info, Pearson_correlation, Cross_Correlation

def data_label(
    file_path,
    window_size = 24,
    n_clusters = 5,
    step_size = 1
    ):
    data = pd.read_csv(file_path)
    subsets = [data.iloc[i:i+window_size] for i in range(0, len(data) - window_size + 1, step_size)]
    flattened_subsets = [np.array(subset).flatten() for subset in subsets]
    kmeans_models = KMeans(n_clusters=n_clusters).fit(flattened_subsets)
    labels = kmeans_models.labels_
    # label_counts = Counter(kmeans_models.labels_)
    # for label, count in label_counts.items():
    #     print(f"Label {label}: {count} times")
    labeled_dataset = [[subsets[i], labels[i]] for i in range(len(subsets))]
    return labeled_dataset


def split(dataset, nr_clients: int, status: str, seed: int) -> List[Subset]:
    rng = npr.default_rng(seed)
    labels = np.array([target for _data, target in dataset])

    if status == 'random':
        splits = np.array_split(rng.permutation(len(dataset)), nr_clients)

    elif status == 'balance_label':
        unique_labels, label_counts = np.unique(labels, return_counts=True)
        label_proportions = label_counts / label_counts.sum()
        samples_per_label_per_client = np.floor(label_proportions * len(dataset) / nr_clients).astype(int)
        splits = [[] for _ in range(nr_clients)]
        for label, samples_per_client in zip(unique_labels, samples_per_label_per_client):
            indices_with_label = np.where(labels == label)[0]
            rng.shuffle(indices_with_label)
            split_indices = np.array_split(indices_with_label, nr_clients)
            for client, indices in enumerate(split_indices):
                splits[client].extend(indices.tolist())
        for split in splits:
            rng.shuffle(split)

    elif status == 'non_iid':
        sorted_indices = np.argsort(np.array([target for _data, target in dataset]))
        shards = np.array_split(sorted_indices, 2 * nr_clients)
        shuffled_shard_indices = rng.permutation(len(shards))
        splits = [
            np.concatenate([shards[i] for i in inds], dtype=np.int64)
            for inds in shuffled_shard_indices.reshape(-1, 2)
        ]

    return [Subset(dataset, split) for split in cast(List[List[int]], splits)]

labeled_dataset = data_label(file_path = 'Data/datasets/stock_data.csv', step_size = 1)
sample_split = split(labeled_dataset, 5, 'non_iid', 42)

statistic, p_value = KS_test(sample_split)
print(f'KS test - statistic: {statistic} - p_value: {p_value}')

average_mi, total_mi_inside = compute_mutual_info(sample_split)
print(f"Average Mutual Information between all subsets: {average_mi}")
# print(f"Average Mutual Information inside each subset: {total_mi_inside}")

correlations, p_values = Pearson_correlation(sample_split)
print(f"Pearson correlation - correlations: {correlations} - p_values: {p_values}")

cross_corrs, grangers_p_values = Cross_Correlation(sample_split)
print(f"Cross Correlation: {cross_corrs} - Granger p_values: {grangers_p_values}")