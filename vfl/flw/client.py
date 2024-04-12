import flwr as fl
import torch
from collections import OrderedDict

from Models.vae.vae import Autoencoder, CustomLoss


class FlowerClient(fl.client.NumPyClient):
    def __init__(self, cid, data):
        super(FlowerClient, self).__init__()
        self.cid = cid
        self.data = data
        self.model = Autoencoder(D_in=data.shape[-1]).cuda()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        self.loss_fn = CustomLoss()
        self.embedding = self.model(self.data)

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.embedding = self.model(self.data)
        return self.get_parameters(config={}), len(self.data), {}

    def evaluate(self, parameters, config):
        # self.model.zero_grad()
        # self.embedding.backward(torch.from_numpy(parameters[int(self.cid)]))
        # self.optimizer.step()
        # return 0.0, 1, {}
        pass
