import sys
import warnings

sys.path.append("../../")
warnings.filterwarnings("ignore")

import argparse
import torch
import flwr as fl

from fedavg import get_fedavg_fn
from fedmultiavg import get_fedmultiavg_fn
from utils import random_cluster_clients

from Utils.io_utils import load_yaml_config, instantiate_from_config


def main():
    """Load model for
    1. server-side parameter initialization
    2. server-side parameter evaluation
    """
    parser = argparse.ArgumentParser(description="Flower")
    parser.add_argument(
        "--num_clients",
        type=int,
        default=1,
        help="Specifies the artificial data partition.",
    )
    parser.add_argument(
        "--num_clusters",
        type=int,
        default=1,
        help="Specifies number of clusters for clients.",
    )
    parser.add_argument(
        "--config_file",
        type=str,
        default=None,
        help="path of config file",
    )
    parser.add_argument("--multi_avg", action="store_true", help="use FedAvg strategy")
    args = parser.parse_args()

    config = load_yaml_config(args.config_file)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = instantiate_from_config(config["model"]).to(device)
    model_parameters = [val.cpu().numpy() for _, val in model.state_dict().items()]

    if args.multi_avg:
        client_clusters = random_cluster_clients(args.num_clients, args.num_clusters)
        strategy = get_fedmultiavg_fn(model_parameters, client_clusters)
    else:
        strategy = get_fedavg_fn(model_parameters)

    # Start Flower server for four rounds of federated learning
    fl.server.start_server(
        server_address="localhost:2424",
        config=fl.server.ServerConfig(num_rounds=2),
        strategy=strategy,
    )


if __name__ == "__main__":
    main()
