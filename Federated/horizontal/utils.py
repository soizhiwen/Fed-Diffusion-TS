import math
import csv
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


def get_cluster_id(client_id, client_clusters):
    client_id = int(client_id)
    for idx, cluster in enumerate(client_clusters):
        if client_id in cluster:
            return idx
    return -1


def partition_clients(num_clients, num_clusters, save_dir=None, seed=42):
    assert (
        num_clusters <= num_clients
    ), "Number of clusters should be less than number of clients"
    rng = npr.default_rng(seed)
    arr = rng.permutation(num_clients)
    arr = np.array_split(arr, num_clusters)
    arr = [a.tolist() for a in arr]

    if save_dir:
        df = pd.DataFrame(arr, dtype=pd.Int64Dtype())
        df.to_csv(f"{save_dir}/client_clusters.csv", header=False, index=False)
    return arr


def partition_features(num_feats, num_partitions, full_ratio=0.2, save_dir=None):
    if num_partitions == 1:
        return None

    def generate_features_group(seed):
        npr.seed(seed)
        num_gen = npr.randint(2, num_feats)
        return npr.choice(num_feats, num_gen, replace=False)

    num_not_random = 1 if full_ratio == 0 else math.ceil(num_partitions * full_ratio)

    features_groups = []
    for i in range(num_partitions - num_not_random):
        features_groups.append(generate_features_group(i))

    if full_ratio == 0:
        concat = np.concatenate(features_groups, axis=0)
        unique = np.unique(concat)
        not_selected_feats = np.setdiff1d(np.arange(num_feats), unique)

        if not_selected_feats.size > 0:
            features_groups.append(not_selected_feats)
        else:
            features_groups.append(generate_features_group(42))
    else:
        for i in range(num_not_random):
            features_groups.append(np.arange(num_feats))

    features_groups = [a.tolist() for a in features_groups]
    if save_dir:
        df = pd.DataFrame(features_groups, dtype=pd.Int64Dtype())
        df.to_csv(f"{save_dir}/features_groups.csv", header=False, index=False)
    return features_groups


def partition_data(
    dataset, nr_clients: int, split_type: str, seed: int
) -> List[Subset]:
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


def load_data_partitions(
    path,
    partition_id,
    nr_clients=5,
    split_type="balance_label",
    seed=42,
):
    ds = np.load(path, allow_pickle=True)
    part_ds = partition_data(ds, nr_clients, split_type, seed)[partition_id]
    part_ds = np.concatenate(part_ds[:, 0])
    return part_ds


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
        select_idx = npr.randint(low=0, high=size, size=(num_select,))
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


def write_csv(fields, name, save_dir):
    with open(f"{save_dir}/{name}.csv", "a") as f:
        writer = csv.writer(f)
        writer.writerow(fields)


def plot_metrics(history, strategy, save_dir):
    m_name = {
        "train_loss": "Train Loss",
        "all_context_fid": "Context-FID Score",
        "all_cross_corr": "Correlational Score",
        "exist_context_fid": "Context-FID Score",
        "exist_cross_corr": "Correlational Score",
    }
    metrics = defaultdict(list)

    if strategy in ["fedavg", "fedweightedavg"]:
        for m, values in history.metrics_distributed_fit.items():
            for r, v in values:
                metrics[m].append((r, v))

        for m, values in history.metrics_distributed.items():
            for r, v in values:
                metrics[m].append((r, v))

        for k, v in metrics.items():
            df = pd.DataFrame(v, columns=["Round", m_name[k]])
            ax = sns.lineplot(data=df, x="Round", y=m_name[k], markers=True, seed=42)
            _ = ax.set_xticks(df["Round"].unique())
            _ = ax.set_xlabel("Round")
            _ = ax.set_ylabel(m_name[k])
            df.to_csv(f"{save_dir}/{k}.csv", index=False)
            plt.savefig(f"{save_dir}/{k}.pdf", bbox_inches="tight")
            plt.clf()

    elif strategy == "fedmultiavg":
        for cluster_id, rounds in history.metrics_distributed_fit.items():
            for r, m in rounds:
                metrics["Train Loss"].append((r, m["train_loss"], cluster_id))

        for cluster_id, rounds in history.metrics_distributed.items():
            for r, m in rounds:
                metrics["Context-FID Score"].append((r, m["context_fid"], cluster_id))
                metrics["Correlational Score"].append((r, m["cross_corr"], cluster_id))

        for k, v in metrics.items():
            df = pd.DataFrame(v, columns=["Round", k, "Cluster"])
            ax = sns.lineplot(
                data=df, x="Round", y=k, hue="Cluster", markers=True, seed=42
            )
            sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
            _ = ax.set_xticks(df["Round"].unique())
            _ = ax.set_xlabel("Round")
            _ = ax.set_ylabel(k)
            name = k.lower().replace(" ", "_")
            df.to_csv(f"{save_dir}/{name}.csv", index=False)
            plt.savefig(f"{save_dir}/{name}.pdf", bbox_inches="tight")
            plt.clf()
    else:
        raise NotImplementedError()

    clients_csv = [(f"clients_{k}", k) for k in metrics]

    for csv, k in clients_csv:
        df = pd.read_csv(f"{save_dir}/{csv}.csv", header=None)
        df.columns = ["Round", m_name[k], "Client ID"]
        df.sort_values(by=["Client ID", "Round"], inplace=True)
        ax = sns.lineplot(
            data=df, x="Round", y=m_name[k], hue="Client ID", markers=True, seed=42
        )
        sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
        _ = ax.set_xticks(df["Round"].unique())
        _ = ax.set_xlabel("Round")
        _ = ax.set_ylabel(m_name[k])
        df.to_csv(f"{save_dir}/{csv}.csv", index=False)
        plt.savefig(f"{save_dir}/{csv}.pdf", bbox_inches="tight")
        plt.clf()
