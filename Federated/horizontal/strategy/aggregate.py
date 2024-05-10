from functools import reduce
from typing import List, Tuple

import numpy as np

from flwr.common import FitRes, NDArrays, parameters_to_ndarrays
from flwr.server.client_proxy import ClientProxy


def weighted_aggregate_inplace(
    results: List[Tuple[ClientProxy, FitRes]],
    features_groups: List[Tuple[int]],
    num_features_total: int,
) -> NDArrays:
    """Compute in-place weighted average."""
    # Count total examples
    num_examples_total = sum(fit_res.num_examples for (_, fit_res) in results)

    # Compute scaling factors for each result
    scaling_factors = []
    for client, fit_res in results:
        # Get the feature group of the client
        exclude_feats = features_groups[int(client.cid)]
        # Get the number of features of the client
        num_features = len(exclude_feats)
        # Compute the scaling factor
        scale_1 = fit_res.num_examples / num_examples_total
        scale_2 = num_features / num_features_total
        scaling_factors.append(scale_1 * scale_2)

    # Let's do in-place aggregation
    # Get first result, then add up each other
    params = [
        scaling_factors[0] * x for x in parameters_to_ndarrays(results[0][1].parameters)
    ]
    for i, (_, fit_res) in enumerate(results[1:]):
        res = (
            scaling_factors[i + 1] * x
            for x in parameters_to_ndarrays(fit_res.parameters)
        )
        params = [reduce(np.add, layer_updates) for layer_updates in zip(params, res)]

    return params
