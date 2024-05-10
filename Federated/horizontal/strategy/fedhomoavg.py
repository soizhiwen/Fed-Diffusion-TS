from collections import defaultdict
from logging import WARNING
from typing import Callable, Dict, List, Optional, Tuple, Union, Any

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
)
from flwr.common.logger import log
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy.aggregate import aggregate_inplace, weighted_loss_avg
from flwr.server.strategy.fedavg import FedAvg

from Federated.horizontal.strategy.utils import *


class FedHomoAvg(FedAvg):
    def __init__(
        self,
        features_groups: List[Tuple[int]],
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
    ) -> None:
        super().__init__(
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            min_fit_clients=min_fit_clients,
            min_evaluate_clients=min_evaluate_clients,
            min_available_clients=min_available_clients,
            evaluate_fn=evaluate_fn,
            on_fit_config_fn=on_fit_config_fn,
            on_evaluate_config_fn=on_evaluate_config_fn,
            accept_failures=accept_failures,
            initial_parameters=initial_parameters,
            fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
            evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
        )
        self.features_groups = features_groups

    def __repr__(self) -> str:
        """Compute a string representation of the strategy."""
        rep = f"FedHomoAvg(accept_failures={self.accept_failures})"
        return rep

    def cluster_clients(
        self, results: List[Tuple[ClientProxy, Any]]
    ) -> Dict[Tuple, List[Tuple[ClientProxy, Any]]]:
        clustered_clients = defaultdict(list)
        for client, res in results:
            features = self.features_groups[int(client.cid)]
            clustered_clients[features].append((client, res))
        return clustered_clients

    def initialize_parameters(
        self, client_manager: ClientManager
    ) -> Dict[Tuple[int], Optional[Parameters]]:
        """Initialize global model parameters."""
        initial_parameters = {}
        for features in set(self.features_groups):
            initial_parameters[features] = self.initial_parameters
        self.initial_parameters = None  # Don't keep initial parameters in memory
        return initial_parameters

    def configure_fit(
        self,
        server_round: int,
        parameters: Dict[Tuple[int], Parameters],
        client_manager: ClientManager,
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure the next round of training."""
        config = {}
        if self.on_fit_config_fn is not None:
            # Custom fit config function provided
            config = self.on_fit_config_fn(server_round)

        # Sample clients
        sample_size, min_num_clients = self.num_fit_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )

        client_pairs = []
        for client in clients:
            features = self.features_groups[int(client.cid)]
            fit_ins = FitIns(parameters[features], config)
            client_pairs.append((client, fit_ins))

        # Return client/config pairs
        return client_pairs

    def configure_evaluate(
        self,
        server_round: int,
        parameters: Dict[Tuple[int], Parameters],
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
        sample_size, min_num_clients = self.num_evaluation_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )

        client_pairs = []
        for client in clients:
            features = self.features_groups[int(client.cid)]
            evaluate_ins = EvaluateIns(parameters[features], config)
            client_pairs.append((client, evaluate_ins))

        # Return client/config pairs
        return client_pairs

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Dict[Tuple[int], Optional[Parameters]], Dict[Tuple[int], Dict[str, Scalar]]]:
        """Aggregate fit results using weighted average."""
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        clustered_clients = self.cluster_clients(results)

        # Does in-place weighted average of results
        parameters_aggregated = {}
        for k, v in clustered_clients.items():
            aggregated_ndarrays = aggregate_inplace(v)
            parameters_aggregated[k] = ndarrays_to_parameters(aggregated_ndarrays)

        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            for k, v in clustered_clients.items():
                fit_metrics = [(res.num_examples, res.metrics) for _, res in v]
                metrics_aggregated[k] = self.fit_metrics_aggregation_fn(fit_metrics)
        elif server_round == 1:  # Only log this warning once
            log(WARNING, "No fit_metrics_aggregation_fn provided")

        return parameters_aggregated, metrics_aggregated

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Dict[Tuple[int], Optional[float]], Dict[Tuple[int], Dict[str, Scalar]]]:
        """Aggregate evaluation losses using weighted average."""
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        clustered_clients = self.cluster_clients(results)

        # Aggregate loss
        loss_aggregated = {}
        for k, v in clustered_clients.items():
            loss = [(res.num_examples, res.loss) for _, res in v]
            loss_aggregated[k] = weighted_loss_avg(loss)

        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        if self.evaluate_metrics_aggregation_fn:
            for k, v in clustered_clients.items():
                eval_metrics = [(res.num_examples, res.metrics) for _, res in v]
                metrics_aggregated[k] = self.evaluate_metrics_aggregation_fn(
                    eval_metrics
                )
        elif server_round == 1:  # Only log this warning once
            log(WARNING, "No evaluate_metrics_aggregation_fn provided")

        return loss_aggregated, metrics_aggregated


def get_fedhomoavg_fn(
    model_parameters,
    features_groups,
    *,
    fraction_fit=1.0,
    fraction_evaluate=1.0,
    min_fit_clients=2,
    min_evaluate_clients=2,
    min_available_clients=2,
):
    strategy = FedHomoAvg(
        features_groups=features_groups,
        fraction_fit=fraction_fit,
        fraction_evaluate=fraction_evaluate,
        min_fit_clients=min_fit_clients,
        min_evaluate_clients=min_evaluate_clients,
        min_available_clients=min_available_clients,
        on_fit_config_fn=fit_config,
        on_evaluate_config_fn=evaluate_config,
        initial_parameters=ndarrays_to_parameters(model_parameters),
        fit_metrics_aggregation_fn=fit_weighted_average,
        evaluate_metrics_aggregation_fn=evaluate_weighted_average,
    )
    return strategy
