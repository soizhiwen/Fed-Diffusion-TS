from functools import reduce
from logging import WARNING
from typing import Callable, Dict, List, Optional, Tuple, Union
from collections import defaultdict
import numpy as np

from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    MetricsAggregationFn,
    NDArrays,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.common.logger import log
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy.aggregate import weighted_loss_avg
from flwr.server.strategy.strategy import Strategy

from Federated.horizontal.utils import get_cluster_id


WARNING_MIN_AVAILABLE_CLIENTS_TOO_LOW = """
Setting `min_available_clients` lower than `min_fit_clients` or
`min_evaluate_clients` can cause the server to fail when there are too few clients
connected to the server. `min_available_clients` must be set to a value larger
than or equal to the values of `min_fit_clients` and `min_evaluate_clients`.
"""


class FedMultiAvg(Strategy):
    def __init__(
        self,
        client_clusters: List[List[int]],
        *,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
        evaluate_fn: Optional[
            Callable[
                [int, NDArrays, Dict[str, Scalar]],
                Optional[Tuple[float, Dict[str, Scalar]]],
            ]
        ] = None,
        on_fit_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        on_evaluate_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        accept_failures: bool = True,
        initial_parameters: Optional[Parameters] = None,
        fit_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
        evaluate_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
        inplace: bool = True,
    ) -> None:
        super().__init__()

        if (
            min_fit_clients > min_available_clients
            or min_evaluate_clients > min_available_clients
        ):
            log(WARNING, WARNING_MIN_AVAILABLE_CLIENTS_TOO_LOW)

        self.client_clusters = client_clusters
        self.fraction_fit = fraction_fit
        self.fraction_evaluate = fraction_evaluate
        self.min_fit_clients = min_fit_clients
        self.min_evaluate_clients = min_evaluate_clients
        self.min_available_clients = min_available_clients
        self.evaluate_fn = evaluate_fn
        self.on_fit_config_fn = on_fit_config_fn
        self.on_evaluate_config_fn = on_evaluate_config_fn
        self.accept_failures = accept_failures
        self.initial_parameters = initial_parameters
        self.fit_metrics_aggregation_fn = fit_metrics_aggregation_fn
        self.evaluate_metrics_aggregation_fn = evaluate_metrics_aggregation_fn
        self.inplace = inplace

    def __repr__(self) -> str:
        """Compute a string representation of the strategy."""
        rep = f"FedMultiAvg(accept_failures={self.accept_failures})"
        return rep

    def __num_fit_clients(self, num_available_clients: int) -> Tuple[int, int]:
        """Return the sample size and the required number of available clients."""
        num_clients = int(num_available_clients * self.fraction_fit)
        return max(num_clients, self.min_fit_clients), self.min_available_clients

    def __num_evaluation_clients(self, num_available_clients: int) -> Tuple[int, int]:
        """Use a fraction of available clients for evaluation."""
        num_clients = int(num_available_clients * self.fraction_evaluate)
        return max(num_clients, self.min_evaluate_clients), self.min_available_clients

    def __cluster_results(self, results):
        cluster_results = defaultdict(list)
        for client, fit_res in results:
            cluster_id = get_cluster_id(client.cid, self.client_clusters)
            cluster_results[cluster_id].append((client, fit_res))
        return cluster_results

    def __aggregate(
        self, results: List[Tuple[NDArrays, int]], len_model_params: int
    ) -> NDArrays:
        """Compute weighted average."""
        # Calculate the total number of examples used during training
        num_examples_total = sum(num_examples for (_, num_examples) in results)

        # Create a list of weights, each multiplied by the related number of examples
        model_weighted_weights = [
            [layer * num_examples for layer in weights[:len_model_params]]
            for weights, num_examples in results
        ]

        # Compute average weights of each layer
        model_weights_prime: NDArrays = [
            reduce(np.add, layer_updates) / num_examples_total
            for layer_updates in zip(*model_weighted_weights)
        ]

        # Aggregate EMA params
        ema_weighted_weights = [
            [layer * num_examples for layer in weights[len_model_params:]]
            for weights, num_examples in results
        ]

        # Compute average weights of each layer
        ema_weights_prime: NDArrays = [
            reduce(np.add, layer_updates) / num_examples_total
            for layer_updates in zip(*ema_weighted_weights)
        ]
        return model_weights_prime + ema_weights_prime

    def __aggregate_inplace(
        self, results: List[Tuple[ClientProxy, FitRes]], cluster_id: int
    ) -> NDArrays:
        """Compute in-place weighted average."""
        # Count total examples
        num_examples_total = sum(fit_res.num_examples for (_, fit_res) in results)

        # Compute scaling factors for each result
        scaling_factors = [
            fit_res.num_examples / num_examples_total for _, fit_res in results
        ]

        # Let's do in-place aggregation
        # Get first result, then add up each other
        len_model_params = results[0][1].metrics["len_model_params"]
        self.len_model_params_clusters[cluster_id] = len_model_params
        model_params = [
            scaling_factors[0] * x
            for x in parameters_to_ndarrays(results[0][1].parameters)[:len_model_params]
        ]
        for i, (_, fit_res) in enumerate(results[1:]):
            res = (
                scaling_factors[i + 1] * x
                for x in parameters_to_ndarrays(fit_res.parameters)[:len_model_params]
            )
            model_params = [
                reduce(np.add, layer_updates)
                for layer_updates in zip(model_params, res)
            ]

        # Aggregate EMA params
        ema_params = [
            scaling_factors[0] * x
            for x in parameters_to_ndarrays(results[0][1].parameters)[len_model_params:]
        ]
        for i, (_, fit_res) in enumerate(results[1:]):
            res = (
                scaling_factors[i + 1] * x
                for x in parameters_to_ndarrays(fit_res.parameters)[len_model_params:]
            )
            ema_params = [
                reduce(np.add, layer_updates) for layer_updates in zip(ema_params, res)
            ]

        return model_params + ema_params

    def __partial_aggregate_inplace(
        self, results: List[Tuple[ClientProxy, FitRes]]
    ) -> NDArrays:
        """Compute in-place weighted average."""
        # Count total examples
        num_examples_total = sum(fit_res.num_examples for (_, fit_res) in results)

        # Compute scaling factors for each result
        scaling_factors = [
            fit_res.num_examples / num_examples_total for _, fit_res in results
        ]

        # Let's do in-place aggregation
        # Get first result, then add up each other
        len_model_params = results[0][1].metrics["len_model_params"]
        params = [
            scaling_factors[0] * x
            for x in parameters_to_ndarrays(results[0][1].parameters)[:len_model_params]
        ]
        for i, (_, fit_res) in enumerate(results[1:]):
            res = (
                scaling_factors[i + 1] * x
                for x in parameters_to_ndarrays(fit_res.parameters)[:len_model_params]
            )
            params = [
                reduce(np.add, layer_updates) for layer_updates in zip(params, res)
            ]

        return params

    def initialize_parameters(
        self, client_manager: ClientManager
    ) -> Dict[int, Optional[Parameters]]:
        """Initialize global model parameters."""
        self.len_model_params_clusters = {}
        len_model_params = len(self.initial_parameters)

        initial_parameters = {}
        self.initial_parameters = ndarrays_to_parameters(self.initial_parameters)
        for idx in range(len(self.client_clusters)):
            self.len_model_params_clusters[idx] = len_model_params
            initial_parameters[idx] = self.initial_parameters
        self.initial_parameters = None  # Don't keep initial parameters in memory
        return initial_parameters

    def evaluate(
        self, server_round: int, parameters: Parameters
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        """Evaluate model parameters using an evaluation function."""
        if self.evaluate_fn is None:
            # No evaluation function provided
            return None
        parameters_ndarrays = parameters_to_ndarrays(parameters)
        eval_res = self.evaluate_fn(server_round, parameters_ndarrays, {})
        if eval_res is None:
            return None
        loss, metrics = eval_res
        return loss, metrics

    def configure_fit(
        self,
        server_round: int,
        parameters: Dict[int, Parameters],
        client_manager: ClientManager,
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure the next round of training."""
        config = {}
        if self.on_fit_config_fn is not None:
            # Custom fit config function provided
            config = self.on_fit_config_fn(server_round)

        # Sample clients
        sample_size, min_num_clients = self.__num_fit_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )

        client_pairs = []
        for client in clients:
            cluster_id = get_cluster_id(client.cid, self.client_clusters)
            config["len_model_params"] = self.len_model_params_clusters[cluster_id]
            fit_ins = FitIns(parameters[cluster_id], config)
            client_pairs.append((client, fit_ins))

        # Return client/config pairs
        return client_pairs

    def configure_evaluate(
        self,
        server_round: int,
        parameters: Dict[int, Parameters],
        client_manager: ClientManager,
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        """Configure the next round of evaluation."""
        # Do not configure federated evaluation if fraction eval is 0.
        if self.fraction_evaluate == 0.0:
            return []

        # Parameters and config
        config = {}
        if self.on_evaluate_config_fn is not None:
            # Custom evaluation config function provided
            config = self.on_evaluate_config_fn(server_round)

        # Sample clients
        sample_size, min_num_clients = self.__num_evaluation_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )

        client_pairs = []
        for client in clients:
            cluster_id = get_cluster_id(client.cid, self.client_clusters)
            config["len_model_params"] = self.len_model_params_clusters[cluster_id]
            evaluate_ins = EvaluateIns(parameters[cluster_id], config)
            client_pairs.append((client, evaluate_ins))

        # Return client/config pairs
        return client_pairs

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Dict[int, Optional[Parameters]], Dict[int, Dict[str, Scalar]]]:
        """Aggregate fit results using weighted average."""
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        cluster_results = self.__cluster_results(results)
        cluster_aggregated_ndarrays = {}

        if self.inplace:
            # Does in-place weighted average of results
            for k, v in cluster_results.items():
                cluster_aggregated_ndarrays[k] = self.__aggregate_inplace(v, k)
        else:
            for k, v in cluster_results.items():
                # Convert results
                weights_results = [
                    (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
                    for _, fit_res in v
                ]
                len_model_params = v[0][1].metrics["len_model_params"]
                self.len_model_params_clusters[k] = len_model_params
                cluster_aggregated_ndarrays[k] = self.__aggregate(
                    weights_results, len_model_params
                )

        if server_round % 2 == 0:
            for k in cluster_results:
                for i, _ in enumerate(cluster_results[k]):
                    cluster_results[k][i][1].parameters = ndarrays_to_parameters(
                        cluster_aggregated_ndarrays[k]
                    )
            all_fit_res = sum(cluster_results.values(), [])
            partial_aggregated_ndarrays = self.__partial_aggregate_inplace(all_fit_res)

            for k, v in cluster_aggregated_ndarrays.items():
                len_model_params = self.len_model_params_clusters[k]
                cluster_aggregated_ndarrays[k][:len_model_params] = partial_aggregated_ndarrays    

        cluster_parameters_aggregated = {}
        for k, v in cluster_aggregated_ndarrays.items():
            cluster_parameters_aggregated[k] = ndarrays_to_parameters(v)

        # Aggregate custom metrics if aggregation fn was provided
        cluster_metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            for k, v in sorted(cluster_results.items()):
                fit_metrics = [(res.num_examples, res.metrics) for _, res in v]
                cluster_metrics_aggregated[k] = self.fit_metrics_aggregation_fn(
                    fit_metrics
                )
        elif server_round == 1:  # Only log this warning once
            log(WARNING, "No fit_metrics_aggregation_fn provided")

        return cluster_parameters_aggregated, cluster_metrics_aggregated

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Dict[int, Optional[float]], Dict[int, Dict[str, Scalar]]]:
        """Aggregate evaluation losses using weighted average."""
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        # Aggregate loss
        cluster_results = self.__cluster_results(results)

        cluster_loss_aggregated = {}
        for k, v in sorted(cluster_results.items()):
            cluster_loss_aggregated[k] = weighted_loss_avg(
                [
                    (evaluate_res.num_examples, evaluate_res.loss)
                    for _, evaluate_res in v
                ]
            )

        # Aggregate custom metrics if aggregation fn was provided
        cluster_metrics_aggregated = {}
        if self.evaluate_metrics_aggregation_fn:
            for k, v in sorted(cluster_results.items()):
                eval_metrics = [(res.num_examples, res.metrics) for _, res in v]
                cluster_metrics_aggregated[k] = self.evaluate_metrics_aggregation_fn(
                    eval_metrics
                )
        elif server_round == 1:  # Only log this warning once
            log(WARNING, "No evaluate_metrics_aggregation_fn provided")

        return cluster_loss_aggregated, cluster_metrics_aggregated


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
    return {
        "train_loss": sum(train_loss) / sum(examples),
    }


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


def get_fedmultiavg_fn(
    model_parameters,
    client_clusters,
    *,
    fraction_fit=1.0,
    fraction_evaluate=1.0,
    min_fit_clients=3,
    min_evaluate_clients=3,
    min_available_clients=3,
):
    strategy = FedMultiAvg(
        client_clusters,
        fraction_fit=fraction_fit,
        fraction_evaluate=fraction_evaluate,
        min_fit_clients=min_fit_clients,
        min_evaluate_clients=min_evaluate_clients,
        min_available_clients=min_available_clients,
        on_fit_config_fn=fit_config,
        on_evaluate_config_fn=evaluate_config,
        initial_parameters=model_parameters,
        fit_metrics_aggregation_fn=fit_weighted_average,
        evaluate_metrics_aggregation_fn=evaluate_weighted_average,
    )
    return strategy
