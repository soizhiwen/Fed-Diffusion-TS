import os
import math
import csv
import torch
import pandas as pd
import numpy as np
import numpy.random as npr
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, cast
from collections import defaultdict
from torch.utils.data import Subset

from Utils.context_fid import Context_FID
from Utils.metric_utils import display_scores
from Utils.cross_correlation import CrossCorrelLoss


def partition_features(
    num_feats,
    num_partitions,
    full_ratio=0.2,
    repeat_thold=0.0,
    save_dir=None,
):
    if num_partitions == 1:
        return None

    def generate_features_group(seed):
        npr.seed(seed)
        num_gen = npr.randint(2, num_feats)
        return npr.choice(num_feats, num_gen, replace=False)

    num_not_random = 1 if full_ratio == 0 else math.ceil(num_partitions * full_ratio)

    features_groups = []
    for i in range(num_partitions - num_not_random):
        npr.seed(i)
        if np.random.uniform(0, 1) >= repeat_thold:
            features_groups.append(generate_features_group(i))
        else:
            features_groups.append(generate_features_group(i + 1))

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

    features_groups = [tuple(a) for a in features_groups]
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
        npr.seed(42)
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
        "fit_exist_context_fid": "Context-FID Score",
        "all_context_fid": "Context-FID Score",
        "all_cross_corr": "Correlational Score",
        "exist_context_fid": "Context-FID Score",
        "exist_cross_corr": "Correlational Score",
    }
    metrics = defaultdict(list)

    if strategy in ["fedavg", "fedweightedavg", "feddynaavg"]:
        for m, values in history.metrics_distributed_fit.items():
            for r, v in values:
                metrics[m].append((r, v))

        for m, values in history.metrics_distributed.items():
            for r, v in values:
                metrics[m].append((r, v))

        for k, v in metrics.items():
            df = pd.DataFrame(v, columns=["Round", m_name[k]])
            ax = sns.lineplot(data=df, x="Round", y=m_name[k], marker="o", seed=42)
            _ = ax.set_xticks(df["Round"].unique())
            _ = ax.set_xlabel("Round")
            _ = ax.set_ylabel(m_name[k])
            df.to_csv(f"{save_dir}/{k}.csv", index=False)
            plt.savefig(f"{save_dir}/{k}.pdf", bbox_inches="tight")
            plt.clf()

    elif strategy in ["fednoavg", "fedhomoavg", "fedtsm", "fedacctsm"]:
        for id, rounds in history.metrics_distributed_fit.items():
            for r, m in rounds:
                metrics["train_loss"].append((r, m["train_loss"], id))
                if "fit_exist_context_fid" in m:
                    metrics["fit_exist_context_fid"].append(
                        (r, m["fit_exist_context_fid"], id)
                    )

        for id, rounds in history.metrics_distributed.items():
            for r, m in rounds:
                metrics["all_context_fid"].append((r, m["all_context_fid"], id))
                metrics["all_cross_corr"].append((r, m["all_cross_corr"], id))
                metrics["exist_context_fid"].append((r, m["exist_context_fid"], id))
                metrics["exist_cross_corr"].append((r, m["exist_cross_corr"], id))

        for k, v in metrics.items():
            df = pd.DataFrame(v, columns=["Round", m_name[k], "Cluster ID"])
            df.sort_values(by=["Cluster ID", "Round"], inplace=True)
            ax = sns.lineplot(
                data=df, x="Round", y=m_name[k], hue="Cluster ID", marker="o", seed=42
            )
            sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
            _ = ax.set_xticks(df["Round"].unique())
            _ = ax.set_xlabel("Round")
            _ = ax.set_ylabel(m_name[k])
            df.to_csv(f"{save_dir}/{k}.csv", index=False)
            plt.savefig(f"{save_dir}/{k}.pdf", bbox_inches="tight")
            plt.clf()

            # Compute global average
            df.drop(columns=["Cluster ID"], inplace=True)
            df = df.groupby(["Round"], as_index=False).mean()
            ax = sns.lineplot(data=df, x="Round", y=m_name[k], marker="o", seed=42)
            _ = ax.set_xticks(df["Round"].unique())
            _ = ax.set_xlabel("Round")
            _ = ax.set_ylabel(m_name[k])
            df.to_csv(f"{save_dir}/avg_{k}.csv", index=False)
            plt.savefig(f"{save_dir}/avg_{k}.pdf", bbox_inches="tight")
            plt.clf()

    else:
        raise NotImplementedError()

    clients_csv = [(f"clients_{k}", k) for k in metrics]

    for csv, k in clients_csv:
        df = pd.read_csv(f"{save_dir}/{csv}.csv", header=None)
        df.columns = ["Round", m_name[k], "Client ID"]
        df.sort_values(by=["Client ID", "Round"], inplace=True)
        ax = sns.lineplot(
            data=df, x="Round", y=m_name[k], hue="Client ID", marker="o", seed=42
        )
        sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
        _ = ax.set_xticks(df["Round"].unique())
        _ = ax.set_xlabel("Round")
        _ = ax.set_ylabel(m_name[k])
        df.to_csv(f"{save_dir}/{csv}.csv", index=False)
        plt.savefig(f"{save_dir}/{csv}.pdf", bbox_inches="tight")
        plt.clf()


def increment_path(path, exist_ok=False, sep="", mkdir=False):
    """
    Generates an incremented file or directory path if it exists, with optional mkdir; args: path, exist_ok=False,
    sep="", mkdir=False.

    Example: runs/exp --> runs/exp{sep}2, runs/exp{sep}3, ... etc
    """
    path = Path(path)
    if path.exists() and not exist_ok:
        path, suffix = (
            (path.with_suffix(""), path.suffix) if path.is_file() else (path, "")
        )

        for n in range(2, 9999):
            p = f"{path}{sep}{n}{suffix}"
            if not os.path.exists(p):
                break
        path = Path(p)

    if mkdir:
        path.mkdir(parents=True, exist_ok=True)

    return path.as_posix()
