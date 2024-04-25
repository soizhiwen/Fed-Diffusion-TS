import warnings
import torch
import numpy as np
import numpy.random as npr
from typing import List, cast
from tqdm import tqdm
from torch.utils.data import Subset
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")


def apply_transforms(*, train_set, test_set, valid_set=None):
    """Apply transforms to the partition from FederatedDataset."""
    X_train = np.concatenate(train_set[:, 0])
    X_test = np.concatenate(test_set[:, 0])

    scaler = MinMaxScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    train_set[:, 0] = np.split(X_train, len(train_set))
    test_set[:, 0] = np.split(X_test, len(test_set))

    if valid_set is not None:
        X_valid = np.concatenate(valid_set[:, 0])
        X_valid = scaler.transform(X_valid)
        valid_set[:, 0] = np.split(X_valid, len(valid_set))
        return train_set, valid_set, test_set

    return train_set, test_set


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

    return [np.take(dataset, split, axis=0) for split in cast(List[List[int]], splits)]


def load_partition(
    path,
    partition_id,
    nr_clients=5,
    split_type="balance_label",
    valid_size=0.1,
    test_size=0.2,
    seed=42,
):
    ds = np.load(path, allow_pickle=True)
    part_ds = partition(ds, nr_clients, split_type, seed)[partition_id]
    train_set, test_set = train_test_split(
        part_ds,
        test_size=test_size,
        random_state=seed,
    )
    train_set, valid_set = train_test_split(
        train_set,
        test_size=valid_size,
        random_state=seed,
    )

    train_set, valid_set, test_set = apply_transforms(
        train_set=train_set,
        test_set=test_set,
        valid_set=valid_set,
    )

    return train_set, valid_set, test_set


def load_centralized_data(path, test_size=0.2, seed=42):
    ds = np.load(path, allow_pickle=True)
    train_set, test_set = train_test_split(
        ds,
        test_size=test_size,
        random_state=seed,
    )

    _, test_set = apply_transforms(
        train_set=train_set,
        test_set=test_set,
    )

    return test_set


def train(
    model,
    train_loader,
    valid_loader,
    optimizer,
    loss_fn,
    n_epochs,
    device,
    steps=None,
):
    print("Start training...")
    model.train()
    for _ in tqdm(range(n_epochs), desc="Training"):
        for features, labels in train_loader:
            features = features.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(features)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

    train_loss, train_accuracy = test(model, train_loader, loss_fn, device, steps)
    valid_loss, valid_accuracy = test(model, valid_loader, loss_fn, device, steps)

    return {
        "train_loss": train_loss,
        "train_accuracy": train_accuracy,
        "valid_loss": valid_loss,
        "valid_accuracy": valid_accuracy,
    }


def test(model, test_loader, loss_fn, device, steps=None):
    print("Start evalutation...")
    test_loss = 0.0
    n_correct = 0
    model.eval()
    with torch.no_grad():
        for idx, (features, labels) in enumerate(test_loader):
            features = features.to(device)
            labels = labels.to(device)
            outputs = model(features)
            loss = loss_fn(outputs, labels)
            test_loss += loss.item()
            n_correct += torch.sum(outputs.argmax(1) == labels).item()
            if steps is not None and idx == steps:
                break

    loss = test_loss / len(test_loader)
    accuracy = n_correct / len(test_loader.dataset)
    return loss, accuracy
