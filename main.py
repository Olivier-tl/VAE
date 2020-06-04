import os

import fire
import numpy as np
import torch
from torchvision import datasets, transforms
from torchvision.utils import save_image
from tqdm import tqdm

from vae import VAE
from vae import vae_loss

SEED = 0


def train_epoch(model, optimizer, train_loader, device):
    model.train()
    train_loss = 0
    for data, _ in tqdm(train_loader, desc='train epoch', leave=False):
        data = data.to(device)
        data = data.view(-1, 784)  # Flatten the image into one vector
        optimizer.zero_grad()
        output, mu, logvar = model(data)
        loss = vae_loss(output, data, mu, logvar)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    train_loss /= len(train_loader.dataset)

    return train_loss


def test_epoch(model, optimizer, test_loader, device, epoch):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, (data, _) in enumerate(tqdm(test_loader, desc='test epoch', leave=False)):
            data = data.to(device)
            data = data.view(-1, 784)  # Flatten the image into one vector
            output, mu, logvar = model(data)
            test_loss += vae_loss(output, data, mu, logvar).item()

            if i == 0:
                n = min(data.shape[0], 8)
                comparison = torch.cat([data.view(-1, 1, 28, 28)[:n], output.view(-1, 1, 28, 28)[:n]])
                save_image(comparison.cpu(), 'results/reconstruction_' + str(epoch) + '.png', nrow=n)

    test_loss /= len(test_loader.dataset)

    return test_loss


def main(epochs=100, batch_size=20):

    # Set things
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(SEED)
    if not os.path.exists('resutls'):
        os.mkdir('results')

    # Datasets
    train_loader = torch.utils.data.DataLoader(datasets.MNIST('./data',
                                                              train=True,
                                                              download=True,
                                                              transform=transforms.ToTensor()),
                                               batch_size=batch_size,
                                               shuffle=True)
    test_loader = torch.utils.data.DataLoader(datasets.MNIST('./data', train=False, transform=transforms.ToTensor()),
                                              batch_size=batch_size,
                                              shuffle=True)

    model = VAE().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    print('Training...')
    for epoch in range(epochs):

        train_loss = train_epoch(model, optimizer, train_loader, device)
        test_loss = test_epoch(model, optimizer, test_loader, device, epoch)

        print(f'Epoch: {epoch} - Train Loss: {train_loss}, Test Loss: {test_loss}')


if __name__ == "__main__":
    fire.Fire(main)