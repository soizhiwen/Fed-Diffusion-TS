import flwr as fl
import numpy as np
from strategy import FedAvgStrategy
from client import FlowerClient
from pathlib import Path

# partitions, label = get_partitions_and_label()


def client_fn(cid):
    return FlowerClient(cid, partitions[int(cid)]).to_client()


# Start Flower server
hist = fl.simulation.start_simulation(
    client_fn=client_fn,
    num_clients=3,
    config=fl.server.ServerConfig(num_rounds=1000),
    strategy=FedAvgStrategy(),
)

# results_dir = Path("_static/results")
# results_dir.mkdir(exist_ok=True)
# np.save(str(results_dir / "hist.npy"), hist)