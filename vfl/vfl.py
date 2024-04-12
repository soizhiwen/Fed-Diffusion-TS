from torch import nn
import sys
import os
import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

sys.path.append(os.path.join(os.path.dirname(__file__), "../"))


from Models.vae.vae import Autoencoder, CustomLoss
from Models.interpretable_diffusion.gaussian_diffusion import Diffusion_TS

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class VFL(nn.Module):
    def __init__(self, client_model):
        super(VFL, self).__init__()
        self.num_cli = None
        self.cli_features = None
        self.client_model = client_model
        self.server_model = Diffusion_TS(
            seq_length=24,
            feature_size=9,
            n_layer_enc=2,
            n_layer_dec=2,
            d_model=64,  # 4 X 16
            timesteps=500,
            sampling_timesteps=500,
            loss_type="l1",
            beta_schedule="cosine",
            n_heads=4,
            mlp_hidden_times=4,
            attn_pd=0.0,
            resid_pd=0.0,
            kernel_size=1,
            padding_size=0,
        ).to(device)

    def train_with_settings(
        self, epochs, batch_sz, cli_features, real_data, optimizer, loss_fn
    ):
        self.cli_features = cli_features
        self.optimizer = optimizer
        self.criterion = loss_fn
        train_losses = []

        real = [real_data[:, feats[0] : feats[-1] + 1] for feats in cli_features]
        num_batches = (
            len(real_data) // batch_sz
            if len(real_data) % batch_sz == 0
            else len(real_data) // batch_sz + 1
        )

        for epoch in range(epochs):
            self.optimizer.zero_grad()
            total_loss = 0.0
            for minibatch in range(num_batches):
                if minibatch == num_batches - 1:
                    minibatch_data = [r[int(minibatch * batch_sz) :] for r in real]
                else:
                    minibatch_data = [
                        r[int(minibatch * batch_sz) : int((minibatch + 1) * batch_sz)]
                        for r in real
                    ]

                outs, mu, logvar = self.forward(minibatch_data)
                outs = torch.cat(outs, dim=1)
                minibatch_data = torch.cat(minibatch_data, dim=1)
                loss = self.criterion(outs, minibatch_data, mu, logvar)
                total_loss += loss
                loss.backward()
                self.optimizer.step()

            train_losses.append(total_loss.detach().numpy() / num_batches)
            print(f"Epoch: {epoch} Loss: {train_losses[-1]:.3f}")

        return train_losses

    def forward(self, x):
        x = [
            self.client_model[i](x[i], forward_step="encode")
            for i in range(len(self.client_model))
        ]
        x = torch.cat(x, dim=1)
        x = self.server_model(x)
        x = torch.split(x, [12] * len(self.client_model), dim=1)
        return [
            self.client_model[i](x[i], forward_step="decode")
            for i in range(len(self.client_model))
        ]


if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)

    # Preprocess data
    df = pd.read_csv("../Data/datasets/stock_data.csv")
    scaler = MinMaxScaler()
    data = scaler.fit_transform(df)
    real_data = torch.tensor(data).float().to(device)

    # Split data among clients
    num_clients = 3
    features_per_client = (num_clients - 1) * [real_data.shape[1] // num_clients]
    features_per_client.append(real_data.shape[1] - sum(features_per_client))
    features_per_client = np.array(features_per_client)
    all_feature_names = np.array(range(real_data.shape[1]))
    client_feature_names = []

    start_index = 0
    for num_feats in features_per_client:
        feat_names = all_feature_names[start_index : start_index + num_feats]
        client_feature_names.append(feat_names)
        start_index += num_feats

    # Define models
    outs_per_client = 2
    client_model = [Autoencoder(len(in_feats)).to(device) for in_feats in client_feature_names]

    Network = VFL(client_model)
    optimizer = torch.optim.Adam(Network.parameters(), lr=1e-3)
    loss_mse = CustomLoss()

    # Train model and plot losses
    EPOCHS = 300
    BATCH_SIZE = 64

    train_losses = Network.train_with_settings(
        EPOCHS,
        BATCH_SIZE,
        client_feature_names,
        real_data,
        optimizer,
        loss_mse,
    )

    # plot_losses(EPOCHS, train_losses, "Number of clients", num_clients)
