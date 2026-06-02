import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

class VAEBackbone(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dims=[256, 128]):
        super().__init__()

        # Build Encoder
        encoder_layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            encoder_layers.append(nn.Linear(prev_dim, h_dim))
            encoder_layers.append(nn.ReLU())
            prev_dim = h_dim
        self.encoder_backbone = nn.Sequential(*encoder_layers)

        # Latent space projections
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_logvar = nn.Linear(hidden_dims[-1], latent_dim)

        # Build Decoder
        decoder_layers = []
        prev_dim = latent_dim
        for h_dim in reversed(hidden_dims):
            decoder_layers.append(nn.Linear(prev_dim, h_dim))
            decoder_layers.append(nn.ReLU())
            prev_dim = h_dim
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x):
        hidden = self.encoder_backbone(x)
        return self.fc_mu(hidden), self.fc_logvar(hidden)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar


class VariationalAutoencoder:
    def __init__(self, n_components=2, hidden_dims=[256, 128], lr=1e-3, epochs=30, batch_size=256, device="cpu"):
        self.n_components = n_components
        self.hidden_dims = hidden_dims
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size

        self.device = torch.device(device)
        self.model = None

    def fit_transform(self, X, y=None):
        X_tensor = torch.tensor(X, dtype=torch.float32)
        input_dim = X_tensor.shape[1]

        self.model = VAEBackbone(input_dim, self.n_components, self.hidden_dims).to(self.device)
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        dataset = TensorDataset(X_tensor)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.model.train()
        for epoch in range(self.epochs):
            for batch in dataloader:
                x_batch = batch[0].to(self.device)
                optimizer.zero_grad()

                recon_x, mu, logvar = self.model(x_batch)

                recon_loss = nn.MSELoss(reduction='sum')(recon_x, x_batch)

                kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

                total_loss = (recon_loss + kl_loss) / x_batch.size(0)
                total_loss.backward()
                optimizer.step()

        return self.transform(X)

    def transform(self, X):
        if self.model is None:
            raise ValueError("The VAE model instances must be fitted before calling transform.")

        self.model.eval()
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            mu, _ = self.model.encode(X_tensor)

        return mu.cpu().numpy()