import warnings
import flwr as fl

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


def evaluate_weighted_average(metrics):
    # Multiply context fid of each client by number of examples used
    context_fids = [num_examples * m["context_fid"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"context_fid": sum(context_fids) / sum(examples)}


def get_fedavg_fn(
    model_parameters,
    *,
    fraction_fit=1.0,
    fraction_evaluate=1.0,
    min_fit_clients=2,
    min_evaluate_clients=2,
    min_available_clients=2,
):
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=fraction_fit,
        fraction_evaluate=fraction_evaluate,
        min_fit_clients=min_fit_clients,
        min_evaluate_clients=min_evaluate_clients,
        min_available_clients=min_available_clients,
        on_fit_config_fn=fit_config,
        on_evaluate_config_fn=evaluate_config,
        initial_parameters=fl.common.ndarrays_to_parameters(model_parameters),
        evaluate_metrics_aggregation_fn=evaluate_weighted_average,
    )
    return strategy
