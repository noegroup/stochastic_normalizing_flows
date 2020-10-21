import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import datasets, transforms
from torchvision.utils import save_image

import numpy as np
import math
import time

from flow_models import GenerativeModel, ModelEval, SimpleVAE, RealNVPVAE, LangevinVAE, SNFVAE

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

def train(model_name, data_file, M):
    start = time.process_time()
    
    latent_dim = 50
    batch_size = 128
    log_interval = 100

    if data_file == 'mnist_data':
        train_loader = torch.utils.data.DataLoader(
            datasets.MNIST('mnist_data', train=True, download=True,
                           transform=transforms.ToTensor()),
                            batch_size=batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(
            datasets.MNIST('mnist_data', train=False, transform=transforms.ToTensor()),
            batch_size=batch_size, shuffle=True)
    else:
        train_loader = torch.utils.data.DataLoader(
            datasets.FashionMNIST('fashionmnist_data', train=True, download=True,
                           transform=transforms.ToTensor()),
                            batch_size=batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(
            datasets.FashionMNIST('fashionmnist_data', train=False, transform=transforms.ToTensor()),
            batch_size=batch_size, shuffle=True)
    
    if model_name in ['SimpleVAE','RealNVPVAE','LangevinVAE']:
        n_epochs = 40
        if model_name == 'SimpleVAE':
            flow = SimpleVAE(latent_dim).to(device)
        if model_name == 'RealNVPVAE':
            flow = RealNVPVAE(latent_dim).to(device)
        if model_name == 'LangevinVAE':
            flow = LangevinVAE(latent_dim).to(device)
        optim = torch.optim.Adam(flow.parameters(), lr=1e-3)
        #perform training
        for epoch in range(1, n_epochs + 1):
            train_loss = 0
            for batch_idx, (data, _) in enumerate(train_loader):
                data = ((torch.rand_like(data) <= data) + 0.).float()
                data = data.to(device)
                loss = -flow.log_likelihood(data, M).mean()
                optim.zero_grad()
                loss.backward()
                train_loss += loss.item() * len(data)
                optim.step()
                if batch_idx % log_interval == 0:
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch, batch_idx * len(data), len(train_loader.dataset),
                        100. * batch_idx / len(train_loader),
                        loss.item() * len(data) / len(data)))

            test_loss = 0
            for i, (data, _) in enumerate(test_loader):
                data = ((torch.rand_like(data) <= data) + 0.).float()
                data = data.to(device)
                loss = -flow.log_likelihood(data, M).sum()
                test_loss += loss.item()

            test_loss /= len(test_loader.dataset)
            print('====> Test set loss: {:.4f}'.format(test_loss))
    else:
        flow = SNFVAE(latent_dim, nsteps=10, stepsize=1e-2).to(device)
        optim = torch.optim.Adam(flow.parameters(), lr=1e-3)
        n_epochs = 20
        flow_disable = True
        for epoch in range(1, n_epochs + 1):
            train_loss = 0
            for batch_idx, (data, _) in enumerate(train_loader):
                data = ((torch.rand_like(data) <= data) + 0.).float()
                data = data.to(device)
                loss = -flow.log_likelihood(data, M, flow_disable).mean()
                optim.zero_grad()
                loss.backward()
                train_loss += loss.item()
                optim.step()
                if batch_idx % log_interval == 0:
                    print(flow.stepsize_para_list.mean())
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch, batch_idx * len(data), len(train_loader.dataset),
                        100. * batch_idx / len(train_loader),
                        loss.item()))

            test_loss = 0
            for i, (data, _) in enumerate(test_loader):
                data = ((torch.rand_like(data) <= data) + 0.).float()
                data = data.to(device)
                loss = -flow.log_likelihood(data, M, flow_disable).sum()
                test_loss += loss.item()

            test_loss /= len(test_loader.dataset)
            print('====> Test set loss: {:.4f}'.format(test_loss))

        optim = torch.optim.Adam(flow.parameters(), lr=1e-3)
        flow_disable = False
        n_epochs = 20
        flow_disable = True
        for epoch in range(1, n_epochs + 1):
            train_loss = 0
            for batch_idx, (data, _) in enumerate(train_loader):
                data = ((torch.rand_like(data) <= data) + 0.).float()
                data = data.to(device)
                loss = -flow.log_likelihood(data, M, flow_disable).mean()
                optim.zero_grad()
                loss.backward()
                train_loss += loss.item()
                optim.step()
                if batch_idx % log_interval == 0:
                    print(flow.stepsize_para_list.mean())
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch+20, batch_idx * len(data), len(train_loader.dataset),
                        100. * batch_idx / len(train_loader),
                        loss.item()))

            test_loss = 0
            for i, (data, _) in enumerate(test_loader):
                data = ((torch.rand_like(data) <= data) + 0.).float()
                data = data.to(device)
                loss = -flow.log_likelihood(data, M, flow_disable).sum()
                test_loss += loss.item()

            test_loss /= len(test_loader.dataset)
            print('====> Test set loss: {:.4f}'.format(test_loss))


    #calculate the marginal log-likelihood
    loss = ModelEval(flow.G, 2000, data_file)

    print('Running time: %s Seconds'%(time.process_time()-start))

if __name__ == '__main__':
    M = 5
    for model in ['SimpleVAE', 'RealNVPVAE', 'LangevinVAE', 'SNFVAE']:
        for data_file in ['mnist_data', 'fashionmnist_data']:
            train(model, data_file, M)
