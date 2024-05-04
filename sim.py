import os
import warnings
import argparse
import torch
import flwr as fl

from Federated.horizontal.fedavg import get_fedavg_fn
from Federated.horizontal.fedmultiavg import get_fedmultiavg_fn
from Federated.horizontal.client import get_client_fn
from Federated.horizontal.utils import random_cluster_clients, plot_metrics

from Utils.io_utils import load_yaml_config, instantiate_from_config

warnings.filterwarnings("ignore")


def parse_args():
    parser = argparse.ArgumentParser(description="Flower")
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Name of the experiment",
    )
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
        "--num_rounds",
        type=int,
        default=1,
        help="Specifies number of global rounds.",
    )
    parser.add_argument(
        "--config_file",
        type=str,
        default=None,
        help="Path of config file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./OUTPUT",
        help="Directory to save the results",
    )
    parser.add_argument(
        "--multi_avg",
        action="store_true",
        help="Use FedAvg strategy",
    )
    parser.add_argument(
        "--split_type",
        type=str,
        default="balance_label",
        help="Type of data partitioning",
        choices=["balance_label", "random", "non_iid", "order"],
    )
    parser.add_argument(
        "--num_cpus",
        type=int,
        default=2,
        help="Number of CPUs to assign to a virtual client",
    )
    parser.add_argument(
        "--num_gpus",
        type=float,
        default=0.5,
        help="Ratio of GPU memory to assign to a virtual client",
    )

    args = parser.parse_args()
    args.save_dir = os.path.join(args.output, f"{args.name}")
    os.makedirs(args.save_dir, exist_ok=True)
    return args


def main():
    args = parse_args()
    config = load_yaml_config(args.config_file)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = instantiate_from_config(config["model"]).to(device)
    model_parameters = [val.cpu().numpy() for _, val in model.state_dict().items()]

    client_fn = get_client_fn(config, args, model)

    if args.multi_avg:
        client_clusters = random_cluster_clients(args.num_clients, args.num_clusters)
        strategy = get_fedmultiavg_fn(model_parameters, client_clusters)
    else:
        strategy = get_fedavg_fn(model_parameters)

    client_resources = {
        "num_cpus": args.num_cpus,
        "num_gpus": args.num_gpus,
    }

    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=args.num_clients,
        client_resources=client_resources,
        config=fl.server.ServerConfig(num_rounds=args.num_rounds),
        strategy=strategy,
    )

    plot_metrics(history, args.save_dir)


if __name__ == "__main__":
    main()
