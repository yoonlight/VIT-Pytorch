"""
Reference:
- https://github.com/debtanu177/CVAE_MNIST/blob/master/train_cvae.py
"""
import os
import random
import argparse

import yaml
import torch
import torch.nn.functional as F
import numpy as np

from tqdm import tqdm
from torch.utils.data.dataloader import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from load_vae import VAE
from dataset.mnist_color_texture_dataset import MnistDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def loss_function(x, recon, mu, logvar, reduction='sum'):
    """
    Loss function for VAE, combining reconstruction loss and KL divergence.

    :param x: Original input tensor
    :param recon: Reconstructed output tensor
    :param mu: Mean from the latent space
    :param logvar: Log variance from the latent space
    :param reduction: Reduction method for the loss ('sum' or 'mean')
    :return: Tuple of reconstruction loss and KL divergence
    """
    recon_loss = F.mse_loss(recon, x, reduction=reduction)
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss, kld


def train_for_one_epoch(
    epoch_idx, model: VAE, mnist_loader: DataLoader, optimizer: torch.optim.Optimizer
):
    r"""
    Method to run the training for one epoch.

    :param epoch_idx: iteration number of current epoch
    :param model: Transformer model
    :param mnist_loader: Data loder for mnist
    :param optimizer: optimizer to be used taken from config
    :return:
    """
    losses = []
    criterion = torch.nn.CrossEntropyLoss()
    for data in tqdm(mnist_loader):
        im = data['image'].float().to(device)
        # number_cls = data['number_cls'].long().to(device)
        optimizer.zero_grad()
        recon, mu, logvar = model(im)
        recon_loss, kld = loss_function(im, recon, mu, logvar)
        loss = recon_loss + kld
        losses.append(loss.item())
        loss.backward()
        optimizer.step()
    print(
        f'Finished epoch: {epoch_idx + 1} | Number Loss : {np.mean(losses):.4f}',
        file=open(os.path.join('train_vae.txt'), 'a', encoding='utf-8')
    )
    return np.mean(losses)


def train(args):
    #  Read the config file
    ######################################
    with open(args.config_path, 'r', encoding='utf-8') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    print(
        config,
        file=open(os.path.join('train_vae.txt'), 'a', encoding='utf-8')
    )
    #######################################

    # Set the desired seed value
    ######################################
    seed = config['train_params']['seed']
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if device == 'cuda':
        torch.cuda.manual_seed_all(args.seed)
    #######################################

    # Create the model and dataset
    model = VAE(latent_dim=128, im_channels=3, device=device).to(device)
    mnist = MnistDataset('train', config['dataset_params'],
                         im_h=config['model_params']['image_height'],
                         im_w=config['model_params']['image_width'])
    mnist_loader = DataLoader(
        mnist, batch_size=config['train_params']['batch_size'], shuffle=True, num_workers=4)
    num_epochs = config['train_params']['epochs']
    optimizer = Adam(model.parameters(), lr=config['train_params']['lr'])
    scheduler = ReduceLROnPlateau(
        optimizer, factor=0.5, patience=2, verbose=True)

    # Create output directories
    if not os.path.exists(config['train_params']['task_name']):
        os.mkdir(config['train_params']['task_name'])

    # Load checkpoint if found
    if os.path.exists(os.path.join(config['train_params']['task_name'],
                                   config['train_params']['ckpt_name'])):
        print('Loading checkpoint')
        load_path = os.path.join(config['train_params']['task_name'],
                                 config['train_params']['ckpt_name'])
        model.load_state_dict(torch.load(load_path, map_location=device))
    best_loss = np.inf

    for epoch_idx in range(num_epochs):
        mean_loss = train_for_one_epoch(
            epoch_idx, model, mnist_loader, optimizer)
        scheduler.step(mean_loss)
        # Simply update checkpoint if found better version
        if mean_loss < best_loss:
            print(
                f'Improved Loss to {mean_loss:.4f} .... Saving Model',
                file=open(os.path.join('train_vae.txt'), 'a', encoding='utf-8')
            )
            save_path = os.path.join(config['train_params']['task_name'],
                                     config['train_params']['ckpt_name'])
            torch.save(model.state_dict(), save_path)
            best_loss = mean_loss
        else:
            print('No Loss Improvement')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for vit training')
    parser.add_argument('--config', dest='config_path',
                        default='config/vae.yaml', type=str)
    args = parser.parse_args()
    train(args)
