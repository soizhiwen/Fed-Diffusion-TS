def fit_config(server_round: int):
    """Return training configuration dict for each round."""
    config = {
        "size_every": 2001,
        "metric_iterations": 5,
        "local_epochs": 2500,
        "server_round": server_round,
    }
    return config


def evaluate_config(server_round: int):
    """Return evaluation configuration dict for each round."""
    config = {
        "size_every": 2001,
        "metric_iterations": 5,
        "server_round": server_round,
    }
    return config


def fit_weighted_average(metrics):
    # Multiply context fid of each client by number of examples used
    train_loss = [num_examples * m["train_loss"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    agg_metrics = {"train_loss": sum(train_loss) / sum(examples)}

    if "fit_exist_context_fid" in metrics[0][1]:
        fit_exist_context_fid = [
            num_examples * m["fit_exist_context_fid"] for num_examples, m in metrics
        ]
        agg_metrics["fit_exist_context_fid"] = sum(fit_exist_context_fid) / sum(
            examples
        )

    # Aggregate and return custom metric (weighted average)
    return agg_metrics


def evaluate_weighted_average(metrics):
    # Multiply context fid of each client by number of examples used
    all_context_fids = [
        num_examples * m["all_context_fid"] for num_examples, m in metrics
    ]
    all_cross_corrs = [
        num_examples * m["all_cross_corr"] for num_examples, m in metrics
    ]
    examples = [num_examples for num_examples, _ in metrics]

    agg_metrics = {
        "all_context_fid": sum(all_context_fids) / sum(examples),
        "all_cross_corr": sum(all_cross_corrs) / sum(examples),
    }

    if "exist_context_fid" in metrics[0][1] and "exist_cross_corr" in metrics[0][1]:
        exist_context_fids = [
            num_examples * m["exist_context_fid"] for num_examples, m in metrics
        ]
        exist_cross_corrs = [
            num_examples * m["exist_cross_corr"] for num_examples, m in metrics
        ]
        agg_metrics["exist_context_fid"] = sum(exist_context_fids) / sum(examples)
        agg_metrics["exist_cross_corr"] = sum(exist_cross_corrs) / sum(examples)

    # Aggregate and return custom metric (weighted average)
    return agg_metrics
