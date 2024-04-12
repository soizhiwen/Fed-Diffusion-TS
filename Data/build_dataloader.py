import torch
from Utils.io_utils import instantiate_from_config


def build_dataloader(config, args=None):
    batch_size = config["dataloader"]["batch_size"]
    jud = config["dataloader"]["shuffle"]
    config["dataloader"]["train_dataset"]["params"]["output_dir"] = args.save_dir
    dataset = instantiate_from_config(config["dataloader"]["train_dataset"])

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=jud,
        num_workers=0,
        pin_memory=True,
        sampler=None,
        drop_last=jud,
    )

    dataload_info = {"dataloader": dataloader, "dataset": dataset}

    return dataload_info


def build_dataloader_cond(config, args=None):
    batch_size = config["dataloader"]["sample_size"]
    config["dataloader"]["test_dataset"]["params"]["output_dir"] = args.save_dir
    if args.mode == "infill":
        config["dataloader"]["test_dataset"]["params"][
            "missing_ratio"
        ] = args.missing_ratio
    elif args.mode == "predict":
        config["dataloader"]["test_dataset"]["params"]["predict_length"] = args.pred_len
    test_dataset = instantiate_from_config(config["dataloader"]["test_dataset"])

    dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        sampler=None,
        drop_last=False,
    )

    dataload_info = {"dataloader": dataloader, "dataset": test_dataset}

    return dataload_info


def build_dataloader_fed(config, num_clients, args=None):
    batch_size = config["dataloader"]["batch_size"]
    jud = config["dataloader"]["shuffle"]
    config["dataloader"]["train_dataset"]["params"]["output_dir"] = args.save_dir
    dataset = instantiate_from_config(config["dataloader"]["train_dataset"])

    # Split dataset into `num_clients` partitions to simulate different local datasets
    num_features = dataset[0].shape[-1]
    features_per_client = (num_clients - 1) * [num_features // num_clients]
    features_per_client.append(num_features - sum(features_per_client))

    datasets = [[] for _ in range(num_clients)]
    for ds in dataset:
        split = torch.split(ds, features_per_client, dim=1)
        for i, s in enumerate(split):
            datasets[i].append(s)

    # TODO: Split dataset.__dict__
    dataload_infos = []
    for ds in datasets:
        dataloader = torch.utils.data.DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=jud,
            num_workers=0,
            pin_memory=True,
            sampler=None,
            drop_last=jud,
        )

        dataload_infos.append(
            {
                "dataloader": dataloader,
                "dataset": dataset,
                "feature_size": ds[0].shape[-1],
            }
        )

    return dataload_infos


if __name__ == "__main__":
    pass
