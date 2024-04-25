import warnings
import torch
import flwr as fl
from typing import Dict, Optional, Tuple
from collections import OrderedDict
from torch.utils.data import DataLoader

from model import Net
from custom_dataset import CustomDataset
from utils import load_centralized_data, test

warnings.filterwarnings("ignore")


def fit_config(server_round: int):
    """Return training configuration dict for each round.

    Keep batch size fixed at 32, perform two rounds of training with one local epoch,
    increase to two local epochs afterwards.
    """
    config = {
        "batch_size": 16,
        "local_epochs": 1 if server_round < 2 else 2,
    }
    return config


def evaluate_config(server_round: int):
    """Return evaluation configuration dict for each round.

    Perform five local evaluation steps on each client (i.e., use five batches) during
    rounds one to three, then increase to ten local evaluation steps.
    """
    valid_steps = 5 if server_round < 4 else 10
    return {"valid_steps": valid_steps}


def get_evaluate_fn(model, loss_fn, device):
    """Return an evaluation function for server-side evaluation."""

    # Load data here to avoid the overhead of doing it in `evaluate` itself
    centralized_data = load_centralized_data(
        "../../Data/datasets/labeled_stock_data.npy"
    )

    centralized_dataset = CustomDataset(centralized_data)
    centralized_loader = DataLoader(centralized_dataset, batch_size=16)

    # The `evaluate` function will be called after every round
    def evaluate(
        server_round: int,
        parameters: fl.common.NDArrays,
        config: Dict[str, fl.common.Scalar],
    ) -> Optional[Tuple[float, Dict[str, fl.common.Scalar]]]:
        # Update model with the latest parameters
        params_dict = zip(model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        model.load_state_dict(state_dict, strict=True)

        loss, accuracy = test(model, centralized_loader, loss_fn, device)
        return loss, {"accuracy": accuracy}

    return evaluate


def main():
    """Load model for
    1. server-side parameter initialization
    2. server-side parameter evaluation
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = Net()
    model_parameters = [val.cpu().numpy() for _, val in model.state_dict().items()]
    loss_fn = torch.nn.CrossEntropyLoss()

    # Create strategy
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=2,
        min_evaluate_clients=2,
        min_available_clients=10,
        evaluate_fn=get_evaluate_fn(model, loss_fn, device),
        on_fit_config_fn=fit_config,
        on_evaluate_config_fn=evaluate_config,
        initial_parameters=fl.common.ndarrays_to_parameters(model_parameters),
    )

    # Start Flower server for four rounds of federated learning
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=4),
        strategy=strategy,
    )


if __name__ == "__main__":
    main()
