import os
import torch
import flwr as fl
import numpy as np
from collections import OrderedDict

from Federated.horizontal.utils import (
    get_cluster_id,
    load_partition,
    cal_context_fid,
    cal_cross_corr,
)

from engine.solver import Trainer
from Data.build_dataloader import build_dataloader_fed
from Models.interpretable_diffusion.model_utils import unnormalize_to_zero_to_one


class FlowerClient(fl.client.NumPyClient):
    def __init__(self, trainer: Trainer):
        self.trainer = trainer
        self.model = trainer.model

    def get_parameters(self):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        # Update local model parameters
        self.set_parameters(parameters)

        # Get hyperparameters for this round
        self.trainer.train_num_steps = config["local_epochs"]
        train_loss = self.trainer.train()

        parameters_prime = self.get_parameters()
        dataset = self.trainer.dataloader_info["dataset"]

        results = {"train_loss": float(train_loss)}

        return parameters_prime, len(dataset), results

    def evaluate(self, parameters, config):
        """Evaluate parameters on the locally held test set."""
        # Update local model parameters
        self.set_parameters(parameters)

        # Get config values
        size_every = config["size_every"]
        metric_iterations = config["metric_iterations"]

        dataset = self.trainer.dataloader_info["dataset"]
        seq_length, feature_dim = dataset.window, dataset.var_num
        ori_data = np.load(
            os.path.join(
                dataset.dir, f"{dataset.name}_norm_truth_{seq_length}_train.npy"
            )
        )
        fake_data = self.trainer.sample(
            num=len(dataset), size_every=size_every, shape=[seq_length, feature_dim]
        )
        if dataset.auto_norm:
            fake_data = unnormalize_to_zero_to_one(fake_data)
            np.save(
                os.path.join(
                    self.trainer.args.save_dir, f"ddpm_fake_{dataset.name}.npy"
                ),
                fake_data,
            )

        ctx_fid_mean = cal_context_fid(
            ori_data, fake_data, iterations=metric_iterations
        )
        cross_corr_mean = cal_cross_corr(
            ori_data, fake_data, iterations=metric_iterations
        )

        metrics = {
            "context_fid": float(ctx_fid_mean),
            "cross_corr": float(cross_corr_mean),
        }

        return 0.0, len(dataset), metrics


def get_client_fn(
    config, args, model, exclude_feats_clusters=None, client_clusters=None
):
    """Return a function to construct a client.

    The VirtualClientEngine will execute this function whenever a client is sampled by
    the strategy to participate.
    """

    def client_fn(cid: str) -> fl.client.Client:
        """Construct a FlowerClient with its own dataset partition."""

        # Get the partition corresponding to the i-th client
        args.client_id = int(cid)
        dataset = load_partition(
            config["dataloader"]["train_dataset"]["params"]["data_root"],
            args.client_id,
            nr_clients=args.num_clients,
            split_type=args.split_type,
        )

        exclude_feats = None
        if exclude_feats_clusters is not None:
            cluster_id = get_cluster_id(args.client_id, client_clusters)
            exclude_feats = exclude_feats_clusters[cluster_id]

        model.exclude_feats = exclude_feats
        dataloader_info = build_dataloader_fed(config, dataset, args, exclude_feats)
        trainer = Trainer(
            config=config,
            args=args,
            model=model,
            dataloader=dataloader_info,
        )

        # Create and return client
        return FlowerClient(trainer=trainer).to_client()

    return client_fn
