from logging import WARNING
from typing import Callable, Dict, List, Optional, Tuple, Union

from flwr.common import (
    FitRes,
    MetricsAggregationFn,
    NDArrays,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
)
from flwr.common.logger import log
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy.fedavg import FedAvg

from Federated.horizontal.strategy.aggregate import weighted_aggregate_inplace
from Federated.horizontal.strategy.utils import *


class FedDynaAvg(FedAvg):
    def __init__(
        self,
        features_groups: List[Tuple[int]],
        num_features_total: int,
        num_rounds: int,
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
        self.num_features_total = num_features_total
        self.num_rounds = num_rounds

        len_feats_groups = []
        for idx, group in enumerate(features_groups):
            len_feats_groups.append((len(group), idx))
        self.sorted_len_feats_groups = sorted(len_feats_groups, reverse=True)

    def __repr__(self) -> str:
        """Compute a string representation of the strategy."""
        rep = f"FedDynaAvg(accept_failures={self.accept_failures})"
        return rep

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate fit results using weighted average."""
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        k = (1 - (server_round / self.num_rounds)) * self.num_features_total
        top_k_clients = []
        for len_group, client_id in self.sorted_len_feats_groups:
            if len_group >= k:
                top_k_clients.append(client_id)
            else:
                break

        if not top_k_clients:
            first_client = self.sorted_len_feats_groups[0][1]
            top_k_clients.append(first_client)

        top_k_results = []
        for client, fit_res in results:
            if int(client.cid) in top_k_clients:
                top_k_results.append((client, fit_res))

        # Does in-place weighted average of results
        aggregated_ndarrays = weighted_aggregate_inplace(
            top_k_results, self.features_groups, self.num_features_total
        )
        parameters_aggregated = ndarrays_to_parameters(aggregated_ndarrays)

        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
        elif server_round == 1:  # Only log this warning once
            log(WARNING, "No fit_metrics_aggregation_fn provided")

        return parameters_aggregated, metrics_aggregated


def get_feddynaavg_fn(
    model_parameters,
    features_groups,
    num_features_total,
    num_rounds,
    *,
    fraction_fit=1.0,
    fraction_evaluate=1.0,
    min_fit_clients=2,
    min_evaluate_clients=2,
    min_available_clients=2,
):
    strategy = FedDynaAvg(
        features_groups=features_groups,
        num_features_total=num_features_total,
        num_rounds=num_rounds,
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
