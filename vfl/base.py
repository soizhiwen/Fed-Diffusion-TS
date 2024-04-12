import torch
from torch.utils.data import DataLoader

data_path_str = "./data"
ETA = "\N{GREEK SMALL LETTER ETA}"
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
torch.backends.cudnn.deterministic = True


import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Optimizer
from typing import cast
import numpy as np
import numpy.random as npr
from torch.utils.data import Subset
from dataclasses import asdict, dataclass, field
from pandas import DataFrame

from abc import ABC, abstractmethod
from Models.vae.vae import Autoencoder, CustomLoss
from Models.interpretable_diffusion.gaussian_diffusion import Diffusion_TS


class Client(ABC):
    def __init__(self, client_data: Subset, batch_size: int) -> None:
        self.model = Autoencoder(D_in=client_data.shape[-1]).to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        self.loss_fn = CustomLoss()
        self.loader_train = DataLoader(
            client_data,
            batch_size=batch_size,
            shuffle=True,
            drop_last=False,
            generator=self.generator,
        )

    @abstractmethod
    def update(self, weights: list[torch.Tensor], seed: int) -> list[torch.Tensor]: ...


#


class Server(ABC):
    def __init__(self, lr: float, batch_size: int, seed: int) -> None:
        self.clients: list[Client]
        self.lr = lr
        self.batch_size = batch_size
        self.seed = seed
        torch.manual_seed(seed)
        self.model = Diffusion_TS(
            seq_length=24,
            feature_size=feature_size,
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

    @abstractmethod
    def run(self, nr_rounds: int) -> RunResult: ...

    def test(self) -> float:
        correct = 0
        self.model.eval()

        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = self.model(data)
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        return 100.0 * correct / len(cast(datasets.MNIST, test_loader.dataset))


#

from time import perf_counter

from torch.optim import SGD
from tqdm import tqdm


class DecentralizedServer(Server):
    def __init__(
        self,
        lr: float,
        batch_size: int,
        client_subsets: list[Subset],
        client_fraction: float,
        seed: int,
    ) -> None:
        super().__init__(lr, batch_size, seed)
        self.nr_clients = len(client_subsets)
        self.client_fraction = client_fraction
        self.client_sample_counts = [len(subset) for subset in client_subsets]
        self.nr_clients_per_round = max(1, round(client_fraction * self.nr_clients))
        self.rng = npr.default_rng(seed)


class WeightClient(Client):
    def __init__(
        self, client_data: Subset, lr: float, batch_size: int, nr_epochs: int
    ) -> None:
        super().__init__(client_data, batch_size)
        self.optimizer = SGD(params=self.model.parameters(), lr=lr)
        self.nr_epochs = nr_epochs

    def update(self, weights: list[torch.Tensor], seed: int) -> list[torch.Tensor]:
        with torch.no_grad():
            for client_values, server_values in zip(self.model.parameters(), weights):
                client_values[:] = server_values

        self.generator.manual_seed(seed)

        for _epoch in range(self.nr_epochs):
            train_epoch(self.model, self.loader_train, self.optimizer)

        return [x.detach().cpu().clone() for x in self.model.parameters()]


#


class FedAvgServer(DecentralizedServer):
    def __init__(
        self,
        lr: float,
        batch_size: int,
        client_subsets: list[Subset],
        client_fraction: float,
        nr_local_epochs: int,
        seed: int,
    ) -> None:
        super().__init__(lr, batch_size, client_subsets, client_fraction, seed)
        self.name = "FedAvg"
        self.nr_local_epochs = nr_local_epochs
        self.clients = [
            WeightClient(subset, lr, batch_size, nr_local_epochs)
            for subset in client_subsets
        ]

    def run(self, nr_rounds: int) -> RunResult:
        elapsed_time = 0.0
        run_result = RunResult(
            self.name,
            self.nr_clients,
            self.client_fraction,
            self.batch_size,
            self.nr_local_epochs,
            self.lr,
            self.seed,
        )

        for nr_round in tqdm(range(nr_rounds), desc="Rounds", leave=False):
            setup_start_time = perf_counter()
            self.model.train()
            weights = [x.detach().cpu().clone() for x in self.model.parameters()]
            indices_chosen_clients = self.rng.choice(
                self.nr_clients, self.nr_clients_per_round, replace=False
            )
            chosen_sum_nr_samples = sum(
                self.client_sample_counts[i] for i in indices_chosen_clients
            )
            chosen_adjusted_weights: list[list[torch.Tensor]] = []
            elapsed_time += perf_counter() - setup_start_time
            update_time = 0.0

            for c_i in indices_chosen_clients:
                update_start_time = perf_counter()
                ind = int(c_i)
                client_round_seed = (
                    self.seed + ind + 1 + nr_round * self.nr_clients_per_round
                )
                client_weights = self.clients[ind].update(weights, client_round_seed)
                chosen_adjusted_weights.append(
                    [
                        self.client_sample_counts[ind] / chosen_sum_nr_samples * tens
                        for tens in client_weights
                    ]
                )
                update_time = max(update_time, perf_counter() - update_start_time)

            elapsed_time += update_time
            aggregate_start_time = perf_counter()
            averaged_chosen_weights: list[torch.Tensor] = [
                sum(x) for x in zip(*chosen_adjusted_weights)
            ]

            with torch.no_grad():
                for client_weight, server_parameter in zip(
                    averaged_chosen_weights, self.model.parameters()
                ):
                    server_parameter[:] = client_weight.to(device=device)

            elapsed_time += perf_counter() - aggregate_start_time
            run_result.wall_time.append(round(elapsed_time, 1))
            run_result.message_count.append(
                2 * (nr_round + 1) * self.nr_clients_per_round
            )
            run_result.test_accuracy.append(self.test())

        return run_result
