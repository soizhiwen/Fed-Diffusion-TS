def fit_config(server_round: int):
    """Return training configuration dict for each round."""
    config = {
        "local_epochs": 1000,
    }
    return config


def evaluate_config(server_round: int):
    """Return evaluation configuration dict for each round."""
    config = {
        "size_every": 2001,
        "metric_iterations": 5,
    }
    return config


def fit_weighted_average(metrics):
    # Multiply context fid of each client by number of examples used
    train_loss = [num_examples * m["train_loss"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"train_loss": sum(train_loss) / sum(examples)}


def evaluate_weighted_average(metrics):
    # Multiply context fid of each client by number of examples used
    context_fids = [num_examples * m["context_fid"] for num_examples, m in metrics]
    cross_corrs = [num_examples * m["cross_corr"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {
        "context_fid": sum(context_fids) / sum(examples),
        "cross_corr": sum(cross_corrs) / sum(examples),
    }
