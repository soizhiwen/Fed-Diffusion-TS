from torch import nn, optim
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Autoencoder(nn.Module):
    def __init__(self, D_in, H=50, H2=12, latent_dim=3):

        # Encoder
        super(Autoencoder, self).__init__()
        self.optimizer = None
        self.criterion = None
        self.linear1 = nn.Linear(D_in, H)
        self.lin_bn1 = nn.BatchNorm1d(num_features=H)
        self.linear2 = nn.Linear(H, H2)
        self.lin_bn2 = nn.BatchNorm1d(num_features=H2)
        self.linear3 = nn.Linear(H2, H2)
        self.lin_bn3 = nn.BatchNorm1d(num_features=H2)

        # Latent vectors mu and sigma
        self.fc1 = nn.Linear(H2, latent_dim)
        self.bn1 = nn.BatchNorm1d(num_features=latent_dim)
        self.fc21 = nn.Linear(latent_dim, latent_dim)
        self.fc22 = nn.Linear(latent_dim, latent_dim)

        # Sampling vector
        self.fc3 = nn.Linear(latent_dim, latent_dim)
        self.fc_bn3 = nn.BatchNorm1d(latent_dim)
        self.fc4 = nn.Linear(latent_dim, H2)
        self.fc_bn4 = nn.BatchNorm1d(H2)

        # Decoder
        self.linear4 = nn.Linear(H2, H2)
        self.lin_bn4 = nn.BatchNorm1d(num_features=H2)
        self.linear5 = nn.Linear(H2, H)
        self.lin_bn5 = nn.BatchNorm1d(num_features=H)
        self.linear6 = nn.Linear(H, D_in)
        self.lin_bn6 = nn.BatchNorm1d(num_features=D_in)

        self.relu = nn.ReLU()

    def encode(self, x):
        lin1 = self.relu(self.lin_bn1(self.linear1(x)))
        lin2 = self.relu(self.lin_bn2(self.linear2(lin1)))
        lin3 = self.relu(self.lin_bn3(self.linear3(lin2)))

        fc1 = F.relu(self.bn1(self.fc1(lin3)))

        r1 = self.fc21(fc1)
        r2 = self.fc22(fc1)

        return r1, r2

    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, z):
        fc3 = self.relu(self.fc_bn3(self.fc3(z)))
        fc4 = self.relu(self.fc_bn4(self.fc4(fc3)))

        lin4 = self.relu(self.lin_bn4(self.linear4(fc4)))
        lin5 = self.relu(self.lin_bn5(self.linear5(lin4)))
        return self.lin_bn6(self.linear6(lin5))

    def forward(self, x, forward_step="encode"):
        if forward_step == "encode":
            mu, logvar = self.encode(x)
            x = self.reparameterize(mu, logvar)
            return x, mu, logvar
        else:
            x = self.decode(x)
            return x

    def train_with_settings(self, epochs, batch_sz, real_data, optimizer, loss_fn):
        self.optimizer = optimizer
        self.criterion = loss_fn
        if batch_sz == -1:
            batch_sz = len(real_data)
            print("batch_sz: ", batch_sz)
        num_batches = (
            len(real_data) // batch_sz
            if len(real_data) % batch_sz == 0
            else len(real_data) // batch_sz + 1
        )
        train_loss = []
        for epoch in range(epochs):
            self.optimizer.zero_grad()
            total_loss = 0.0
            for minibatch in range(num_batches):
                if minibatch == num_batches - 1:
                    minibatch_data = real_data[int(minibatch * batch_sz) :]
                else:
                    minibatch_data = real_data[
                        int(minibatch * batch_sz) : int((minibatch + 1) * batch_sz)
                    ]

                outs, mu, logvar = self.forward(minibatch_data, forward_step="encode")
                outs = self.forward(outs, forward_step="decode")
                loss = self.criterion(outs, minibatch_data, mu, logvar)
                total_loss += loss
                loss.backward()
                self.optimizer.step()
            train_loss.append(total_loss.detach().numpy())
            print(f"Epoch: {epoch} Loss: {train_loss[-1]:.3f}")
        return outs, train_loss

    def sample(self, nr_samples, mu, logvar):
        sigma = torch.exp(logvar / 2)
        no_samples = nr_samples
        q = torch.distributions.Normal(mu.mean(axis=0), sigma.mean(axis=0))
        z = q.rsample(sample_shape=torch.Size([no_samples]))
        with torch.no_grad():
            pred = self.decode(z).cpu().numpy()
        pred[:, -1] = np.clip(pred[:, -1], 0, 1)
        pred[:, -1] = np.round(pred[:, -1])
        return pred


class CustomLoss(nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()
        self.mse_loss = nn.MSELoss(reduction="sum")

    def forward(self, x_recon, x, mu, logvar):
        loss_MSE = self.mse_loss(x_recon, x)
        loss_KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return loss_MSE + loss_KLD


def train_vae_alone(EPOCHS=200, BATCH_SIZE=64, latent_dim=3):
    df = pd.read_csv("Data/stock_data.csv")
    sc = StandardScaler()
    normalized_data = sc.fit_transform(df)
    data_tensor = torch.tensor(normalized_data).float()
    D_in = data_tensor.shape[1]
    H = 50
    H2 = 12
    model = Autoencoder(D_in, H, H2, latent_dim)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = CustomLoss()

    outs, train_loss = model.train_with_settings(
        EPOCHS, BATCH_SIZE, data_tensor, optimizer, loss_fn
    )
    visualize(train_loss)

    np_outs = outs.detach().numpy()
    original_data = sc.inverse_transform(np_outs)
    pd.DataFrame(original_data, columns=df.columns).to_csv(
        "Data/stock_data_synthetic.csv", index=False
    )


def visualize(train_loss):
    plt.figure(figsize=(10, 5))
    plt.plot(train_loss, label="Train Loss")
    plt.title("Training Loss per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()


def calculate_mean_differences(BATCH_SIZE, original_csv, synthetic_csv):
    original_df = pd.read_csv(original_csv)
    synthetic_df = pd.read_csv(synthetic_csv)
    if not all(original_df.columns == synthetic_df.columns):
        print("Columns do not match.")
        return None
    mean_differences = {}
    mean_differences["BATCH_SIZE"] = BATCH_SIZE
    for column in original_df.columns:
        diff = abs(synthetic_df[column] - original_df[column])
        mean_diff = format(diff.mean(), ".2f")
        mean_differences[column] = mean_diff

    return mean_differences


if __name__ == "__main__":
    np.random.seed(42)
    torch.manual_seed(42)
    EPOCHS = 2000
    BATCH_SIZE = 64
    latent_dim = 3
    train_vae_alone(EPOCHS, BATCH_SIZE, latent_dim)
    mean_differences = calculate_mean_differences(
        BATCH_SIZE,
        original_csv="Data/stock_data.csv",
        synthetic_csv="Data/stock_data_synthetic.csv",
    )
    print(mean_differences)


################### Demostration of how to use encoder and decoder ####################


# from abc import ABC, abstractmethod


# class Client(ABC):
#     def __init__(self, client_data, batch_size: int) -> None:
#         self.model = Autoencoder().to(device)
#         # self.loader_train = depend on your method


# class WeightClient(Client):
#     def __init__(
#         self, client_data: Subset, lr: float, batch_size: int, nr_epochs: int
#     ) -> None:
#         super().__init__(client_data, batch_size)
#         self.optimizer = SGD(params=self.model.parameters(), lr=lr)
#         self.nr_epochs = nr_epochs

#     # since we are using pretrained autoencoder, no need for update
#     def update(self, weights: list[torch.Tensor], seed: int) -> list[torch.Tensor]:
#         # with torch.no_grad():
#         #     for client_values, server_values in zip(self.model.parameters(), weights):
#         #         client_values[:] = server_values

#         self.generator.manual_seed(seed)

#         for _epoch in range(self.nr_epochs):
#             train_epoch(self.model, self.loader_train, self.optimizer)

#         return [x.detach().cpu().clone() for x in self.model.parameters()]


# class Server(ABC):
#     def __init__(self, lr: float, batch_size: int, seed: int) -> None:
#         self.clients: list[Client]
#         self.lr = lr
#         self.batch_size = batch_size
#         self.seed = seed
#         torch.manual_seed(seed)
#         # Put server model here
#         self.model = DDPM().to(device)

#     @abstractmethod
#     def run(self, nr_rounds: int) -> RunResult: ...


# class DecentralizedServer(Server):
#     def __init__(
#         self,
#         lr: float,
#         batch_size: int,
#         client_subsets: list[Subset],
#         client_fraction: float,
#         seed: int,
#     ) -> None:
#         super().__init__(lr, batch_size, seed)
#         self.nr_clients = len(client_subsets)
#         self.client_fraction = client_fraction
#         self.client_sample_counts = [len(subset) for subset in client_subsets]
#         self.nr_clients_per_round = max(1, round(client_fraction * self.nr_clients))
#         self.rng = npr.default_rng(seed)


# class FedAvgServer(DecentralizedServer):
#     def __init__(
#         self,
#         lr: float,
#         batch_size: int,
#         client_subsets: list[Subset],
#         client_fraction: float,
#         nr_local_epochs: int,
#         seed: int,
#     ) -> None:
#         super().__init__(lr, batch_size, client_subsets, client_fraction, seed)
#         self.name = "FedAvg"
#         self.nr_local_epochs = nr_local_epochs
#         if batch_size == -1:
#             self.clients = [
#                 WeightClient(subset, lr, len(subset), nr_local_epochs)
#                 for subset in client_subsets
#             ]
#         else:
#             self.clients = [
#                 WeightClient(subset, lr, batch_size, nr_local_epochs)
#                 for subset in client_subsets
#             ]

#     def run(self, nr_rounds: int) -> RunResult:
#         elapsed_time = 0.0
#         run_result = RunResult(
#             self.name,
#             self.nr_clients,
#             self.client_fraction,
#             self.batch_size,
#             self.nr_local_epochs,
#             self.lr,
#             self.seed,
#         )

#         # client_fraction -> mask, but now set it to 1
#         for nr_round in tqdm(range(nr_rounds), desc="Rounds", leave=False):
#             setup_start_time = perf_counter()
#             self.model.train()

#             # write training code here:
#             # get you data
#             # ...
#             outs = self.forward(self, x_minibatch)
#             # calculate loss
#             pred = torch.argmax(outs, dim=1)
#             actual = torch.argmax(y_minibatch, dim=1)
#             correct += torch.sum((pred == actual))
#             total += len(actual)
#             loss = self.criterion(outs, y_minibatch)
#             total_loss += loss
#             loss.backward()
#             self.optimizer.step()

#         return run_result

#     def forward(self, x):
#         x, mu, logvar = [
#             client.forward(self, x, forward_step="encode") for client in self.clients
#         ]
#         x = np.concatenate(x)
#         x = ddpm(x)
#         x = [client.forward(self, x, forward_step="decode") for client in self.clients]
#         return
