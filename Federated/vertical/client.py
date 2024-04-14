import os
import sys
import torch
from torch import optim
from sklearn.preprocessing import StandardScaler

sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))

from Models.vae.vae import Autoencoder
from Models.vae.loss import CustomLoss


class Client:
    def __init__(
        self,
        client_id,
        data=None,
        latent=None,
        model=None,
    ):
        self.client_id = client_id
        self.data = data
        self.latent = latent
        self.model = model

    def train(
        self,
        H=50,
        H2=12,
        latent_dim=3,
        lr=1e-3,
        epochs=300,
        batch_sz=64,
        save_dir="./OUTPUT",
    ):
        df = self.data
        sc = StandardScaler()
        normalized_data = sc.fit_transform(df)
        data_tensor = torch.tensor(normalized_data).float()
        D_in = data_tensor.shape[1]
        model = Autoencoder(D_in, H, H2, latent_dim)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        loss_fn = CustomLoss()

        train_loss = model.train_with_settings(
            epochs=epochs,
            batch_sz=batch_sz,
            real_data=data_tensor,
            optimizer=optimizer,
            loss_fn=loss_fn,
            client_id=self.client_id,
            save_dir=save_dir,
        )

    def encode(self, save_dir="./OUTPUT"):
        df = self.data
        sc = StandardScaler()
        normalized_data = sc.fit_transform(df)
        data_tensor = torch.tensor(normalized_data).float()

        D_in = data_tensor.shape[1]
        self.model = Autoencoder(D_in)
        model_path = f"{save_dir}/vae_{self.client_id}.pth"
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        latent, _, _ = self.model.forward(data_tensor, function="encode")
        self.latent = latent.detach().numpy()
