import os
import sys
import numpy as np
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), "../"))

from Federated.vertical.client import Client


def init_clients(num_clients):
    clients = []
    for i in range(num_clients):
        client = Client(i)
        clients.append(client)
    return clients


def split_dataset(path, clients):
    df = pd.read_csv(path)
    num_clients = len(clients)
    num_features = df.shape[1]
    features_per_client = (num_clients - 1) * [num_features // num_clients]
    features_per_client.append(num_features - sum(features_per_client))
    features_per_client = np.array(features_per_client)

    all_feature_names = np.array(range(num_features))
    client_feature_names = []
    start_index = 0
    for i in features_per_client:
        feat_names = all_feature_names[start_index : start_index + i]
        client_feature_names.append(feat_names)
        start_index += i

    split_df = [df.iloc[:, f[0] : f[-1] + 1] for f in client_feature_names]

    for client in clients:
        client.data = split_df[client.client_id]

    return clients


def concat_latent(clients):
    latents = [client.latent for client in clients]
    latents = np.concatenate(latents, axis=1)
    df = pd.DataFrame(latents)
    df.to_csv("./Data/datasets/stock_data_latent.csv", index=False)
