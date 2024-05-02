import sys
import warnings

sys.path.append("../../")
warnings.filterwarnings("ignore")

import os
import argparse
import torch
import flwr as fl
import numpy as np
from collections import OrderedDict

from utils import load_partition, cal_context_fid, cal_cross_correl_loss

from engine.solver import Trainer
from Data.build_dataloader import build_dataloader_fed
from Models.interpretable_diffusion.model_utils import unnormalize_to_zero_to_one
from Utils.io_utils import load_yaml_config, instantiate_from_config


class FlowerClient(fl.client.NumPyClient):
    def __init__(self, trainer):
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
        self.trainer.train()

        parameters_prime = self.get_parameters()
        dataset = self.trainer.dataloader_info["dataset"]

        return parameters_prime, len(dataset), {}

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

        ctx_fid_mean, _ = cal_context_fid(
            ori_data, fake_data, iterations=metric_iterations
        )
        corr_loss_mean, _ = cal_cross_correl_loss(
            ori_data, fake_data, iterations=metric_iterations
        )

        return float(corr_loss_mean), len(dataset), {"context_fid": float(ctx_fid_mean)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Flower")
    parser.add_argument(
        "--num_clients",
        type=int,
        default=1,
        help="Specifies the artificial data partition.",
    )
    parser.add_argument(
        "--client_id",
        type=int,
        default=0,
        help="Select client ID to rufn",
    )
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument(
        "--config_file",
        type=str,
        default=None,
        help="path of config file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./OUTPUT",
        help="directory to save the results",
    )

    args = parser.parse_args()
    args.save_dir = os.path.join(args.output, f"{args.name}")

    os.makedirs(args.save_dir, exist_ok=True)
    config = load_yaml_config(args.config_file)

    dataset = load_partition(
        config["dataloader"]["train_dataset"]["params"]["data_root"],
        args.client_id,
        nr_clients=args.num_clients,
        split_type="balance_label",
        # valid_size=0.1,
        # test_size=0.2,
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = instantiate_from_config(config["model"]).to(device)
    dataloader_info = build_dataloader_fed(config, dataset, args)
    trainer = Trainer(config=config, args=args, model=model, dataloader=dataloader_info)

    # Start Flower client
    client = FlowerClient(trainer=trainer).to_client()
    fl.client.start_client(server_address="localhost:2424", client=client)


if __name__ == "__main__":
    main()
