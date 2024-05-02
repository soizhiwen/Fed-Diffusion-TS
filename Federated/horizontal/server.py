import sys
import warnings

sys.path.append("../../")
warnings.filterwarnings("ignore")

import argparse
import torch
import flwr as fl

from fedavg import get_fedavg_fn

from Utils.io_utils import load_yaml_config, instantiate_from_config


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
    parser.add_argument("--fedavg", action="store_true", help="use FedAvg strategy")
    args = parser.parse_args()

    config = load_yaml_config(args.config_file)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = instantiate_from_config(config["model"]).to(device)
    model_parameters = [val.cpu().numpy() for _, val in model.state_dict().items()]

    if args.fedavg:
        strategy = get_fedavg_fn(model_parameters)
    else:
        raise NotImplementedError("Only FedAvg strategy is supported")

    # Start Flower server for four rounds of federated learning
    fl.server.start_server(
        server_address="localhost:2424",
        config=fl.server.ServerConfig(num_rounds=6),
        strategy=strategy,
    )


if __name__ == "__main__":
    main()
