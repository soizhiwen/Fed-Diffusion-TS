import warnings
import argparse
import torch
import flwr as fl
from collections import OrderedDict
from torch.utils.data import DataLoader

from model import Net
from custom_dataset import CustomDataset
from utils import load_partition, train, test

warnings.filterwarnings("ignore")


class FlowerClient(fl.client.NumPyClient):
    def __init__(
        self,
        *,
        train_set,
        valid_set,
        test_set,
        device,
        model,
        optimizer,
        loss_fn,
    ):
        self.train_set = train_set
        self.valid_set = valid_set
        self.test_set = test_set
        self.device = device
        self.model = model.to(device)
        self.optimizer = optimizer
        self.loss_fn = loss_fn

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        # Update local model parameters
        self.set_parameters(parameters)

        # Get hyperparameters for this round
        batch_size = config["batch_size"]
        epochs = config["local_epochs"]

        train_dataset = CustomDataset(self.train_set)
        valid_dataset = CustomDataset(self.valid_set)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        valid_loader = DataLoader(valid_dataset, batch_size=batch_size)

        results = train(
            self.model,
            train_loader,
            valid_loader,
            self.optimizer,
            self.loss_fn,
            epochs,
            self.device,
        )

        parameters_prime = self.get_parameters()
        num_examples_train = len(train_dataset)

        return parameters_prime, num_examples_train, results

    def evaluate(self, parameters, config):
        """Evaluate parameters on the locally held test set."""
        # Update local model parameters
        self.set_parameters(parameters)

        # Get config values
        steps = config["valid_steps"]

        # Evaluate global model parameters on the local test data and return results
        test_dataset = CustomDataset(self.test_set)
        test_loader = DataLoader(test_dataset, batch_size=16)

        loss, accuracy = test(
            self.model,
            test_loader,
            self.loss_fn,
            self.device,
            steps,
        )
        return float(loss), len(test_dataset), {"accuracy": float(accuracy)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Flower")
    parser.add_argument(
        "--num-clients",
        type=int,
        default=1,
        choices=range(0, 10),
        required=False,
        help="Specifies the artificial data partition of CIFAR10 to be used. \
        Picks partition 0 by default",
    )
    parser.add_argument(
        "--client-id",
        type=int,
        default=0,
        choices=range(0, 10),
        required=False,
        help="Specifies the artificial data partition of CIFAR10 to be used. \
        Picks partition 0 by default",
    )

    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load a subset of CIFAR-10 to simulate the local data partition
    train_set, valid_set, test_set = load_partition(
        "../../Data/datasets/labeled_stock_data.npy",
        args.client_id,
        nr_clients=args.num_clients,
        split_type="balance_label",
        valid_size=0.1,
        test_size=0.2,
    )

    # Start Flower client
    model = Net()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    loss_fn = torch.nn.CrossEntropyLoss()

    client = FlowerClient(
        train_set=train_set,
        valid_set=valid_set,
        test_set=test_set,
        device=device,
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
    ).to_client()
    fl.client.start_client(server_address="127.0.0.1:8080", client=client)


if __name__ == "__main__":
    main()
