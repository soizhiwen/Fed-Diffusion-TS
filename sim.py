import json
import warnings
import argparse
import torch
import flwr as fl

from Federated.horizontal.strategy.fedavg import get_fedavg_fn
from Federated.horizontal.strategy.fedweightedavg import get_fedweightedavg_fn
from Federated.horizontal.strategy.fednoavg import get_fednoavg_fn
from Federated.horizontal.strategy.fedhomoavg import get_fedhomoavg_fn
from Federated.horizontal.strategy.feddynaavg import get_feddynaavg_fn
from Federated.horizontal.strategy.fedtsm import get_fedtsm_fn
from Federated.horizontal.client import get_client_fn
from Federated.horizontal.utils import partition_features, plot_metrics, increment_path

from Utils.io_utils import load_yaml_config, instantiate_from_config

warnings.filterwarnings("ignore")


def parse_args():
    parser = argparse.ArgumentParser(description="Flower")
    parser.add_argument(
        "--name",
        type=str,
        default="exp",
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
        "--full_ratio",
        type=float,
        default=0.2,
        help="Ratio of clients with full features",
    )
    parser.add_argument(
        "--repeat_thold",
        type=float,
        default=0.0,
        help="Probability of repeating features group",
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
        "--strategy",
        type=str,
        default="fedavg",
        choices=[
            "fedavg",
            "fedweightedavg",
            "fednoavg",
            "fedhomoavg",
            "feddynaavg",
            "fedtsm",
        ],
        help="Strategy to use for federated learning",
    )
    parser.add_argument(
        "--split_type",
        type=str,
        default="balance_label",
        choices=["balance_label", "random", "non_iid", "order"],
        help="Type of data partitioning",
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
    args.save_dir = increment_path(f"{args.output}/{args.name}", sep="_", mkdir=True)

    with open(f"{args.save_dir}/args.json", "w") as f:
        json.dump(args.__dict__, f, indent=2)

    return args


def main():
    args = parse_args()
    config = load_yaml_config(args.config_file)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = instantiate_from_config(config["model"]).to(device)
    model_parameters = [val.cpu().numpy() for _, val in model.state_dict().items()]
    args.len_model_params = len(model_parameters)

    if "missing_ratio" in config["dataloader"]["train_dataset"]["params"]:
        args.features_groups = partition_features(
            model.feature_size,
            args.num_clients,
            args.full_ratio,
            args.repeat_thold,
            args.save_dir,
        )

    client_fn = get_client_fn(config, args, model)

    if args.strategy == "fedavg":
        strategy = get_fedavg_fn(
            model_parameters,
            min_fit_clients=args.num_clients,
            min_evaluate_clients=args.num_clients,
            min_available_clients=args.num_clients,
        )

    elif args.strategy == "fedweightedavg":
        strategy = get_fedweightedavg_fn(
            model_parameters,
            args.features_groups,
            model.feature_size,
            min_fit_clients=args.num_clients,
            min_evaluate_clients=args.num_clients,
            min_available_clients=args.num_clients,
        )

    elif args.strategy == "fednoavg":
        strategy = get_fednoavg_fn(
            model_parameters,
            args.num_clients,
            min_fit_clients=args.num_clients,
            min_evaluate_clients=args.num_clients,
            min_available_clients=args.num_clients,
        )

    elif args.strategy == "fedhomoavg":
        strategy = get_fedhomoavg_fn(
            model_parameters,
            args.features_groups,
            min_fit_clients=args.num_clients,
            min_evaluate_clients=args.num_clients,
            min_available_clients=args.num_clients,
        )

    elif args.strategy == "feddynaavg":
        strategy = get_feddynaavg_fn(
            model_parameters,
            args.features_groups,
            model.feature_size,
            args.num_rounds,
            min_fit_clients=args.num_clients,
            min_evaluate_clients=args.num_clients,
            min_available_clients=args.num_clients,
        )

    elif args.strategy == "fedtsm":
        strategy = get_fedtsm_fn(
            model_parameters,
            args.num_clients,
            args.features_groups,
            args.save_dir,
            min_fit_clients=args.num_clients,
            min_evaluate_clients=args.num_clients,
            min_available_clients=args.num_clients,
        )

    else:
        raise NotImplementedError()

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

    plot_metrics(history, args.strategy, args.save_dir)


if __name__ == "__main__":
    main()
