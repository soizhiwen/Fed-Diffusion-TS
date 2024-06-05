from logging import WARNING
from typing import Callable, Dict, List, Optional, Tuple, Union

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
from flwr.server.strategy.fedavg import FedAvg

from Federated.horizontal.strategy.utils import *
from Federated.horizontal.utils import write_csv


class FedAccTSM(FedAvg):
    def __init__(
        self,
        num_clients: int,
        num_features_total: int,
        features_groups: List[Tuple[int]],
        save_dir: str,
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
        self.num_clients = num_clients
        self.num_features_total = num_features_total
        self.features_groups = features_groups
        self.save_dir = save_dir
        self.t_cid = -1
        self.overwrite_cids = []

    def __repr__(self) -> str:
        """Compute a string representation of the strategy."""
        rep = f"FedAccTSM(accept_failures={self.accept_failures})"
        return rep

    def initialize_parameters(
        self, client_manager: ClientManager
    ) -> Dict[int, Optional[Parameters]]:
        """Initialize global model parameters."""
        initial_parameters = {}
        for i in range(self.num_clients):
            initial_parameters[i] = self.initial_parameters
        self.initial_parameters = None  # Don't keep initial parameters in memory
        return initial_parameters

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
        sample_size, min_num_clients = self.num_fit_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )

        client_pairs = []
        for client in clients:
            client_id = int(client.cid)
            config["t_cid"] = -1 if client_id in self.overwrite_cids else self.t_cid

            if self.t_cid != -1:
                params = parameters_to_ndarrays(parameters[client_id])
                t_params = parameters_to_ndarrays(parameters[self.t_cid])
                t_exclude_feats = self.features_groups[self.t_cid]
                concat = list(params) + list(t_params) + [t_exclude_feats]
                parameters[client_id] = ndarrays_to_parameters(concat)

            fit_ins = FitIns(parameters[client_id], config)
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
        sample_size, min_num_clients = self.num_evaluation_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )

        client_pairs = []
        for client in clients:
            evaluate_ins = EvaluateIns(parameters[int(client.cid)], config)
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

        # Convert results to dictionary
        parameters_aggregated = {}
        for client, fit_res in results:
            parameters_aggregated[int(client.cid)] = fit_res.parameters

        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            for client, fit_res in results:
                fit_metrics = [(fit_res.num_examples, fit_res.metrics)]
                metrics_aggregated[int(client.cid)] = self.fit_metrics_aggregation_fn(
                    fit_metrics
                )

            for cid, metrics in metrics_aggregated.items():
                len_group = len(self.features_groups[cid])
                penalty = (len_group / self.num_features_total) ** 4
                metrics_aggregated[cid]["penalty_feats_context_fid"] = (
                    metrics["fit_exist_context_fid"] / penalty
                )
                metrics_aggregated[cid]["feats_context_fid"] = (
                    metrics["fit_exist_context_fid"] / len_group
                )

            self.t_cid = min(
                metrics_aggregated,
                key=lambda k: metrics_aggregated[k]["penalty_feats_context_fid"],
            )
            fields = [server_round, self.t_cid]
            write_csv(fields, "teachers", self.save_dir)

            # Update overwrite_cids
            if server_round > 1 and self.t_cid != -1:
                # Normalize feats_context_fid
                feats_ctx_fid = [
                    metrics_aggregated[k]["feats_context_fid"]
                    for k in metrics_aggregated
                ]
                feats_ctx_fid = np.array(feats_ctx_fid)
                mean = np.mean(feats_ctx_fid)
                std = np.std(feats_ctx_fid)
                norm_ctx_fid = (feats_ctx_fid - mean) / std
                for i, k in enumerate(metrics_aggregated):
                    metrics_aggregated[k]["norm_feats_context_fid"] = norm_ctx_fid[i]

                # Overwrite bad students
                self.overwrite_cids = []
                for client, _ in results:
                    cid = int(client.cid)
                    if cid == self.t_cid:
                        continue

                    cid_norm = metrics_aggregated[cid]["norm_feats_context_fid"]
                    t_cid_norm = metrics_aggregated[self.t_cid][
                        "norm_feats_context_fid"
                    ]
                    diff_norm = abs(t_cid_norm - cid_norm)
                    threshold = abs(t_cid_norm / (server_round * 0.5))
                    if cid_norm > t_cid_norm and diff_norm > threshold:
                        parameters_aggregated[cid] = parameters_aggregated[self.t_cid]
                        self.overwrite_cids.append(cid)

                fields = [server_round, self.overwrite_cids]
                write_csv(fields, "overwrite_cids", self.save_dir)

        elif server_round == 1:  # Only log this warning once
            log(WARNING, "No fit_metrics_aggregation_fn provided")

        return parameters_aggregated, metrics_aggregated

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
        loss_aggregated = {}
        for client, evaluate_res in results:
            loss = [(evaluate_res.num_examples, evaluate_res.loss)]
            loss_aggregated[int(client.cid)] = weighted_loss_avg(loss)

        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        if self.evaluate_metrics_aggregation_fn:
            for client, evaluate_res in results:
                eval_metrics = [(evaluate_res.num_examples, evaluate_res.metrics)]
                metrics_aggregated[int(client.cid)] = (
                    self.evaluate_metrics_aggregation_fn(eval_metrics)
                )
        elif server_round == 1:  # Only log this warning once
            log(WARNING, "No evaluate_metrics_aggregation_fn provided")

        return loss_aggregated, metrics_aggregated


def get_fedacctsm_fn(
    model_parameters,
    num_clients,
    num_features_total,
    features_groups,
    save_dir,
    *,
    fraction_fit=1.0,
    fraction_evaluate=1.0,
    min_fit_clients=2,
    min_evaluate_clients=2,
    min_available_clients=2,
):
    strategy = FedAccTSM(
        num_clients=num_clients,
        num_features_total=num_features_total,
        features_groups=features_groups,
        save_dir=save_dir,
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
