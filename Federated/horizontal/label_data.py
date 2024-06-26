"""
python label_data.py --csv_file ~/Fed-Diffusion-TS/Data/datasets/stock_data.csv --num_clusters 5
python label_data.py --csv_file ~/Fed-Diffusion-TS/Data/datasets/energy_data.csv --num_clusters 6
python label_data.py --csv_file ~/Fed-Diffusion-TS/Data/datasets/ETTh.csv --num_clusters 6 --drop_first
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler


def label_data(path, window_size=24, n_clusters=5, step_size=1, drop_first=False):
    data = pd.read_csv(path)
    if drop_first:
        data.drop(data.columns[0], axis=1, inplace=True)

    subsets = [
        data.iloc[i : i + window_size].values
        for i in range(0, len(data) - window_size + 1, step_size)
    ]
    flattened_subsets = [np.array(subset).flatten() for subset in subsets]
    flattened_subsets = MinMaxScaler().fit_transform(flattened_subsets)
    kmeans_models = KMeans(n_clusters=n_clusters).fit(flattened_subsets)
    labels = kmeans_models.labels_
    labeled_dataset = [[subsets[i], labels[i]] for i in range(len(subsets))]
    return labeled_dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Label data")
    parser.add_argument(
        "--csv_file",
        type=str,
        default=None,
        help="Path of CSV file",
    )
    parser.add_argument(
        "--num_clusters",
        type=int,
        default=5,
        help="stock_data=5, energy_data=6",
    )
    parser.add_argument(
        "--window_size",
        type=int,
        default=24,
        help="Window size for labeling data",
    )
    parser.add_argument(
        "--step_size",
        type=int,
        default=1,
        help="Step size for labeling data",
    )
    parser.add_argument(
        "--drop_first",
        action="store_true",
        default=False,
        help="Drop the first column of the dataset",
    )
    args = parser.parse_args()

    dataset = label_data(
        path=args.csv_file,
        window_size=args.window_size,
        n_clusters=args.num_clusters,
        step_size=args.step_size,
        drop_first=args.drop_first,
    )
    dataset = np.array(dataset, dtype=object)
    path = Path(args.csv_file)
    np.save(f"{path.parent}/labeled_{path.stem}.npy", dataset)
