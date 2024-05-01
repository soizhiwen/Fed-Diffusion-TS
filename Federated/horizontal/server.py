import warnings
import argparse
import torch
import flwr as fl
from typing import Dict, Optional, Tuple
from collections import OrderedDict
from torch.utils.data import DataLoader

from utils import load_centralized_data

from Utils.io_utils import load_yaml_config, instantiate_from_config

warnings.filterwarnings("ignore")


def fit_config(server_round: int):
    """Return training configuration dict for each round."""
    config = {
        "local_epochs": 500 if server_round < 5 else 1500,
    }
    return config


def evaluate_config(server_round: int):
    """Return evaluation configuration dict for each round."""
    config = {
        "size_every": 2001,
        "metric_iterations": 1 if server_round < 5 else 5,
    }
    return config


# def get_evaluate_fn(model, loss_fn, device):
#     """Return an evaluation function for server-side evaluation."""

#     # Load data here to avoid the overhead of doing it in `evaluate` itself
#     centralized_data = load_centralized_data(
#         "../../Data/datasets/labeled_stock_data.npy"
#     )

#     centralized_dataset = CustomDataset(centralized_data)
#     centralized_loader = DataLoader(centralized_dataset, batch_size=16)

#     # The `evaluate` function will be called after every round
#     def evaluate(
#         server_round: int,
#         parameters: fl.common.NDArrays,
#         config: Dict[str, fl.common.Scalar],
#     ) -> Optional[Tuple[float, Dict[str, fl.common.Scalar]]]:
#         # Update model with the latest parameters
#         params_dict = zip(model.state_dict().keys(), parameters)
#         state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
#         model.load_state_dict(state_dict, strict=True)

#         loss, accuracy = test(model, centralized_loader, loss_fn, device)
#         return loss, {"accuracy": accuracy}

#     return evaluate


# def weighted_average(metrics):
#     # Multiply accuracy of each client by number of examples used
#     accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
#     examples = [num_examples for num_examples, _ in metrics]

#     # Aggregate and return custom metric (weighted average)
#     return {"accuracy": sum(accuracies) / sum(examples)}


def main():
    """Load model for
    1. server-side parameter initialization
    2. server-side parameter evaluation
    """
    parser = argparse.ArgumentParser(description="Flower")
    parser.add_argument(
        "--config_file",
        type=str,
        default=None,
        help="path of config file",
    )
    args = parser.parse_args()

    config = load_yaml_config(args.config_file)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = instantiate_from_config(config["model"]).to(device)
    model_parameters = [val.cpu().numpy() for _, val in model.state_dict().items()]

    # Create strategy
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=2,
        min_evaluate_clients=2,
        min_available_clients=2,
        # evaluate_fn=get_evaluate_fn(model, loss_fn, device),
        on_fit_config_fn=fit_config,
        on_evaluate_config_fn=evaluate_config,
        initial_parameters=fl.common.ndarrays_to_parameters(model_parameters),
        # fit_metrics_aggregation_fn=,
        # evaluate_metrics_aggregation_fn=weighted_average,
    )

    # Start Flower server for four rounds of federated learning
    fl.server.start_server(
        server_address="localhost:2424",
        config=fl.server.ServerConfig(num_rounds=6),
        strategy=strategy,
    )


if __name__ == "__main__":
    main()
