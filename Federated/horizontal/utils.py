import torch
import pandas as pd
import numpy as np
import numpy.random as npr
import seaborn as sns
import matplotlib.pyplot as plt
from typing import List, cast
from collections import defaultdict
from torch.utils.data import Subset

from Utils.context_fid import Context_FID
from Utils.metric_utils import display_scores
from Utils.cross_correlation import CrossCorrelLoss


def random_cluster_clients(num_clients, num_clusters, seed=42):
    assert (
        num_clusters <= num_clients
    ), "Number of clusters should be less than number of clients"
    rng = npr.default_rng(seed)
    arr = np.arange(num_clients)
    arr = rng.permutation(arr)
    arr = np.array_split(arr, num_clusters)
    return arr


def partition(dataset, nr_clients: int, split_type: str, seed: int) -> List[Subset]:
    rng = npr.default_rng(seed)
    labels = np.array([target for _data, target in dataset])

    if split_type == "random":
        splits = np.array_split(rng.permutation(len(dataset)), nr_clients)

    elif split_type == "balance_label":
        unique_labels, label_counts = np.unique(labels, return_counts=True)
        label_proportions = label_counts / label_counts.sum()
        samples_per_label_per_client = np.floor(
            label_proportions * len(dataset) / nr_clients
        ).astype(int)
        splits = [[] for _ in range(nr_clients)]
        for label, samples_per_client in zip(
            unique_labels, samples_per_label_per_client
        ):
            indices_with_label = np.where(labels == label)[0]
            rng.shuffle(indices_with_label)
            split_indices = np.array_split(indices_with_label, nr_clients)
            for client, indices in enumerate(split_indices):
                splits[client].extend(indices.tolist())
        for split in splits:
            rng.shuffle(split)

    elif split_type == "non_iid":
        sorted_indices = np.argsort(np.array([target for _data, target in dataset]))
        shards = np.array_split(sorted_indices, 2 * nr_clients)
        shuffled_shard_indices = rng.permutation(len(shards))
        splits = [
            np.concatenate([shards[i] for i in inds], dtype=np.int64)
            for inds in shuffled_shard_indices.reshape(-1, 2)
        ]

    elif split_type == "order":
        splits = np.array_split(np.arange(len(dataset)), nr_clients)

    return [np.take(dataset, split, axis=0) for split in cast(List[List[int]], splits)]


def load_partition(
    path,
    partition_id,
    nr_clients=5,
    split_type="balance_label",
    seed=42,
):
    ds = np.load(path, allow_pickle=True)
    part_ds = partition(ds, nr_clients, split_type, seed)[partition_id]
    part_ds = np.concatenate(part_ds[:, 0])
    return part_ds


def load_centralized_data(path):
    ds = np.load(path, allow_pickle=True)
    ds = np.concatenate(ds[:, 0])
    return ds


def cal_context_fid(ori_data, fake_data, iterations=5):
    context_fid_score = []
    for i in range(iterations):
        context_fid = Context_FID(ori_data[:], fake_data[: ori_data.shape[0]])
        context_fid_score.append(context_fid)
        print(f"Iter {i}: Context-FID={context_fid}")

    mean = display_scores(context_fid_score)
    return mean


def cal_cross_corr(ori_data, fake_data, iterations=5):
    def random_choice(size, num_select=100):
        select_idx = np.random.randint(low=0, high=size, size=(num_select,))
        return select_idx

    x_real = torch.from_numpy(ori_data)
    x_fake = torch.from_numpy(fake_data)

    correlational_score = []
    size = int(x_real.shape[0] / iterations)

    for i in range(iterations):
        real_idx = random_choice(x_real.shape[0], size)
        fake_idx = random_choice(x_fake.shape[0], size)
        corr = CrossCorrelLoss(x_real[real_idx, :, :], name="CrossCorrelLoss")
        loss = corr.compute(x_fake[fake_idx, :, :])
        correlational_score.append(loss.item())
        print(f"Iter {i}: Cross-Correlation={loss.item()}")

    mean = display_scores(correlational_score)
    return mean


def plot_metrics(history, save_dir):
    metrics = defaultdict(list)
    for cluster_id, rounds in history.metrics_distributed.items():
        for r, m in rounds:
            metrics["Context-FID Score"].append((r, m["context_fid"], cluster_id))
            metrics["Correlational Score"].append((r, m["cross_corr"], cluster_id))

    for k, v in metrics.items():
        df = pd.DataFrame(v, columns=["Round", k, "Cluster"])
        ax = sns.lineplot(data=df, x="Round", y=k, hue="Cluster", seed=42)
        sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
        _ = ax.set_xticks(df["Round"].unique())
        _ = ax.set_xlabel("Round")
        _ = ax.set_ylabel(k)
        name = k.lower().replace(" ", "_")
        df.to_csv(f"{save_dir}/{name}.csv", index=False)
        plt.tight_layout()
        plt.savefig(f"{save_dir}/{name}.png", dpi=300)
        plt.clf()
