import torch
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, index):
        features, label = self.dataset[index]
        features = torch.tensor(features, dtype=torch.float32)
        features = features.unsqueeze(dim=0)
        label = torch.tensor(label, dtype=torch.long)
        return features, label

    def __len__(self):
        return len(self.dataset)
