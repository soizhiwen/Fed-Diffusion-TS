import os
import warnings
import torch
import flwr as fl
import numpy as np
from collections import OrderedDict

from Federated.horizontal.utils import (
    get_cluster_id,
    load_data_partitions,
    write_csv,
    cal_context_fid,
    cal_cross_corr,
)

from engine.solver import Trainer
from Utils.metric_utils import visualization
from Data.build_dataloader import build_dataloader_fed
from Models.interpretable_diffusion.model_utils import unnormalize_to_zero_to_one

warnings.filterwarnings("ignore")


class FlowerClient(fl.client.NumPyClient):
    def __init__(self, trainer: Trainer):
        self.trainer = trainer
        self.model = trainer.model
        self.ema = trainer.ema
        self.len_model_params = trainer.args.len_model_params
        self.save_dir = trainer.args.save_dir
        self.client_id = trainer.args.client_id
        self.exclude_feats = self.trainer.args.exclude_feats

    def get_parameters(self):
        model_params = [val.cpu().numpy() for _, val in self.model.state_dict().items()]
        ema_params = [val.cpu().numpy() for _, val in self.ema.state_dict().items()]
        return model_params + ema_params

    def set_parameters(self, parameters):
        model_params = parameters[: self.len_model_params]
        params_dict = zip(self.model.state_dict().keys(), model_params)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

        ema_params = parameters[self.len_model_params :]
        if ema_params:
            params_dict = zip(self.ema.state_dict().keys(), ema_params)
            state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
            self.ema.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        # Update local model parameters
        self.set_parameters(parameters)

        # Get hyperparameters for this round
        server_round = config["server_round"]
        self.trainer.train_num_steps = config["local_epochs"]
        train_loss = self.trainer.train()

        parameters_prime = self.get_parameters()
        dataset = self.trainer.dataloader_info["dataset"]
        results = {"train_loss": float(train_loss)}

        for k, v in results.items():
            cid = (
                f"{self.client_id} {self.exclude_feats}"
                if self.exclude_feats
                else self.client_id
            )
            fields = [server_round, v, cid]
            write_csv(fields, f"clients_{k}", self.save_dir)

        return parameters_prime, len(dataset), results

    def evaluate(self, parameters, config):
        """Evaluate parameters on the locally held test set."""
        # Update local model parameters
        self.set_parameters(parameters)

        # Get config values
        size_every = config["size_every"]
        metric_iterations = config["metric_iterations"]
        server_round = config["server_round"]

        dataset = self.trainer.dataloader_info["dataset"]
        seq_length, feature_dim = dataset.window, dataset.var_num

        ori_data = np.load(
            f"{dataset.dir}/{dataset.name}_norm_truth_{seq_length}_train.npy"
        )

        fake_data = self.trainer.sample(
            num=len(dataset), size_every=size_every, shape=[seq_length, feature_dim]
        )

        if dataset.auto_norm:
            os.makedirs(f"{self.save_dir}/fake", exist_ok=True)
            fake_data = unnormalize_to_zero_to_one(fake_data)
            np.save(
                f"{self.save_dir}/fake/ddpm_fake_{dataset.name}_r{server_round}.npy",
                fake_data,
            )

        analysis_types = ["pca", "tsne", "kernel"]

        # Compute metrics with all features
        for analysis in analysis_types:
            visualization(
                ori_data=ori_data,
                generated_data=fake_data,
                analysis=analysis,
                compare=ori_data.shape[0],
                name=f"{dataset.name}_r{server_round}_all",
                save_dir=self.save_dir,
            )

        all_ctx_fid_mean = cal_context_fid(
            ori_data, fake_data, iterations=metric_iterations
        )
        all_cross_corr_mean = cal_cross_corr(
            ori_data, fake_data, iterations=metric_iterations
        )

        metrics = {
            "all_context_fid": float(all_ctx_fid_mean),
            "all_cross_corr": float(all_cross_corr_mean),
        }

        # Compute metrics with existing features
        if self.exclude_feats:
            exist_ori_data = ori_data[:, :, self.exclude_feats]
            exist_fake_data = fake_data[:, :, self.exclude_feats]

            for analysis in analysis_types:
                visualization(
                    ori_data=exist_ori_data,
                    generated_data=exist_fake_data,
                    analysis=analysis,
                    compare=exist_ori_data.shape[0],
                    name=f"{dataset.name}_r{server_round}_exist",
                    save_dir=self.save_dir,
                )

            exist_ctx_fid_mean = cal_context_fid(
                exist_ori_data, exist_fake_data, iterations=metric_iterations
            )
            exist_cross_corr_mean = cal_cross_corr(
                exist_ori_data, exist_fake_data, iterations=metric_iterations
            )
            metrics["exist_context_fid"] = float(exist_ctx_fid_mean)
            metrics["exist_cross_corr"] = float(exist_cross_corr_mean)

        for k, v in metrics.items():
            cid = (
                f"{self.client_id} {self.exclude_feats}"
                if self.exclude_feats
                else self.client_id
            )
            fields = [server_round, v, cid]
            write_csv(fields, f"clients_{k}", self.save_dir)

        return 0.0, len(dataset), metrics


def get_client_fn(config, args, model):
    """Return a function to construct a client.

    The VirtualClientEngine will execute this function whenever a client is sampled by
    the strategy to participate.
    """

    def client_fn(cid: str) -> fl.client.Client:
        """Construct a FlowerClient with its own dataset partition."""

        # Get the partition corresponding to the i-th client
        args.client_id = int(cid)
        dataset = load_data_partitions(
            config["dataloader"]["train_dataset"]["params"]["data_root"],
            args.client_id,
            nr_clients=args.num_clients,
            split_type=args.split_type,
        )

        if hasattr(args, "features_groups"):
            if args.strategy in ["fedavg", "fedweightedavg", "fednoavg"]:
                args.exclude_feats = args.features_groups[args.client_id]
            elif args.strategy == "fedmultiavg":
                cluster_id = get_cluster_id(args.client_id, args.client_clusters)
                args.exclude_feats = args.features_groups[cluster_id]
        else:
            args.exclude_feats = None

        dataloader_info = build_dataloader_fed(config, dataset, args)
        trainer = Trainer(
            config=config,
            args=args,
            model=model,
            dataloader=dataloader_info,
        )

        # Create and return client
        return FlowerClient(trainer=trainer).to_client()

    return client_fn
