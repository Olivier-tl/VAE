import torch
from torch import nn
from torch.nn import functional as F


class VAE(nn.Module):
    def __init__(self, input_dim=784, distribution_dim=20):
        super(VAE, self).__init__()

        self.layer1 = nn.Linear(input_dim, 400)
        self.layer2_mu = nn.Linear(400, distribution_dim)
        self.layer2_logvar = nn.Linear(400, distribution_dim)
        self.layer3 = nn.Linear(distribution_dim, 400)
        self.layer4 = nn.Linear(400, input_dim)

    def encode(self, x):
        hidden = F.relu(self.layer1(x))
        mu = self.layer2_mu(hidden)
        logvar = self.layer2_logvar(hidden)
        return mu, logvar

    def decode(self, z):
        hidden = F.relu(self.layer3(z))
        output = torch.sigmoid(self.layer4(hidden))
        return output

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)  # TODO : Experiment without 0.5, I don't know why it's there
        epsilon = torch.randn_like(std)  # Draw from normal distrib with the same dim as std
        z = mu + std * epsilon
        return z

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        output = self.decode(z)
        return output, mu, logvar


def vae_loss(y_pred, y_true, mu, logvar):
    BCE = F.binary_cross_entropy(y_pred, y_true, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu**2 - torch.exp(logvar))
    return BCE + KLD
