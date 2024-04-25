from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, index):
        features, label = self.dataset[index]
        return features, label

    def __len__(self):
        return len(self.dataset)
