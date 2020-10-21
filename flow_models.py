import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import time

from torchvision import datasets, transforms
from torchvision.utils import save_image

#The model of the Decoder
class GenerativeModel(nn.Module):

    def __init__(self, latent_dim=50):
        super(GenerativeModel, self).__init__()
        self.latent_dim = latent_dim
        self.net = torch.nn.Sequential(
                    torch.nn.Linear(latent_dim, 1024),
                    torch.nn.ReLU(),
                    torch.nn.Linear(1024, 1024),
                    torch.nn.ReLU(),
                    torch.nn.Linear(1024, 784),
                    torch.nn.Sigmoid()
                    )
        
    def forward(self, x):
        return self.net(x)

    def sample(self, M, N=None):
        device = next(self.parameters()).device
        if N is None:
            x = torch.randn(M, self.latent_dim).to(device)
        else:
            x = torch.randn(M, N, self.latent_dim).to(device)
        return self.forward(x)
    
    def conditional_log_likelihood(self, x, y):
        recon_x = torch.clamp(self.forward(x), 1e-6, 1.-1e-6)
        return torch.log(recon_x) * y + torch.log(1 - recon_x) * (1 - y)
        
class SimpleVAE(nn.Module):

    def __init__(self, latent_dim=50):
        super(SimpleVAE, self).__init__()
        self.latent_dim = latent_dim
        self.G = GenerativeModel(latent_dim)
        self.encoder = torch.nn.Sequential(
                    torch.nn.Linear(784, 1024),
                    torch.nn.ReLU(),
                    torch.nn.Linear(1024, 1024),
                    torch.nn.ReLU(),
                    torch.nn.Linear(1024, latent_dim * 2)
                    )
    
    def forward(self, x, y):
        device = next(self.parameters()).device
        M = x.shape[0]
        N = y.shape[0]
        dW = torch.zeros((M, N, 1)).to(device)
        mean_std = self.encoder(y)
        mean = mean_std[:, :self.latent_dim]
        std = torch.abs(mean_std[:, self.latent_dim:]) + 1e-6
        x1 = x * std + mean
        dW = dW + (x**2).sum(axis=2, keepdims=True) / 2
        dW = dW - (x1**2).sum(axis=2, keepdims=True) / 2
        dW = dW + self.G.conditional_log_likelihood(x1, y).sum(axis=2, keepdims=True)
        dW = dW + torch.log(std).sum(axis=1, keepdims=True)
        return x1, dW

    def log_likelihood(self, y, M):
        device = next(self.parameters()).device
        x0 = torch.randn(M, y.shape[0], self.latent_dim).to(device)
        x, dW = self.forward(x0, y.view(-1, 784))
        return torch.mean(dW, axis=0, keepdims=False)

class LangevinVAE(nn.Module):

    def __init__(self, latent_dim=50, nsteps=30, stepsize=0.01):
        super().__init__()
        self.latent_dim = latent_dim
        self.G = GenerativeModel(latent_dim)
        self.nsteps = nsteps
        stepsize_list = torch.FloatTensor([stepsize,] * nsteps)
        lambda_list = (np.array(range(1,nsteps + 1))/nsteps).tolist()
        lambda_list = torch.FloatTensor(lambda_list)
        self.stepsize_para_list, self.lambda_para_list = self.stepsize_lambda_2_para(stepsize_list, lambda_list)
        self.stepsize_para_list = nn.Parameter(torch.FloatTensor(self.stepsize_para_list), requires_grad=True)
        self.lambda_para_list = nn.Parameter(torch.FloatTensor(self.lambda_para_list), requires_grad=True)
        
    def stepsize_lambda_2_para(self, stepsize_list, lambda_list):
        stepsize_para_list = torch.clamp(torch.abs(stepsize_list), min=1e-6)
        lambda_para_list = lambda_list
        return stepsize_para_list, lambda_para_list
    
    def para_2_stepsize_lambda(self, stepsize_para_list, lambda_para_list):
        stepsize_list = torch.abs(stepsize_para_list) + 1e-6
        lambda_list = lambda_para_list
        return stepsize_list, lambda_list

    def energy_0(self, x, y):
        return (x**2).sum(axis=2, keepdims=True) / 2

    def force_0(self, x, y):
        return -x
    
    def sample_energy_0(self, y, M):
        device = next(self.parameters()).device
        x = torch.randn(M, y.shape[0], self.latent_dim).to(device)
        return x
        
    def energy_1(self, x, y):
        return (x**2).sum(axis=2, keepdims=True) / 2 - self.G.conditional_log_likelihood(x, y).sum(axis=2, keepdims=True)

    def force_1(self, x, y):
        x0 = x.clone().detach().requires_grad_(True)
        e = self.energy_1(x0, y)
        return -torch.autograd.grad(e.sum(), x0, create_graph=True)[0]

    def interpolated_energy(self, x, y, lambda_=1.):
        return self.energy_0(x, y) * (1 - lambda_) + self.energy_1(x, y) * lambda_

    def interpolated_force(self, x, y, lambda_=1.):
        return self.force_0(x, y) * (1 - lambda_) + self.force_1(x, y) * lambda_

    def forward(self, x, y):
        stepsize_list, lambda_list = self.para_2_stepsize_lambda(self.stepsize_para_list, self.lambda_para_list)
        dW = self.energy_0(x, y)
        for i in range(self.nsteps):
            lambda_ = lambda_list[i]
            stepsize = stepsize_list[i]
            # forward step
            x1 = x + stepsize * self.interpolated_force(x, lambda_) + torch.sqrt(2*stepsize) * torch.randn_like(x)
            tmp_dW = self.interpolated_energy(x1, y, lambda_) - self.interpolated_energy(x, y, lambda_)
            A = torch.exp(torch.clamp(-tmp_dW, - math.inf, 0.))
            u = torch.rand_like(A)
            acc = (u <= A).float()
            x = (1 - acc) * x + acc * x1
            dW += acc * tmp_dW
        dW = dW - self.energy_1(x, y)
        return x, dW

    def log_likelihood(self, y, M):
        x0 = self.sample_energy_0(y.view(-1, 784), M)
        x, dW = self.forward(x0, y.view(-1, 784))
        return torch.mean(dW, axis=0, keepdims=False)


class CouplingLayer(nn.Module):
    def __init__(self, input_dim, hid_dim, mask, cond_dim=None, s_tanh_activation=True, smooth_activation=False):
        super().__init__()
        
        if cond_dim is not None:
            total_input_dim = input_dim + cond_dim
        else:
            total_input_dim = input_dim

        self.s_fc1 = nn.Linear(total_input_dim, hid_dim)
        self.s_fc2 = nn.Linear(hid_dim, hid_dim)
        self.s_fc3 = nn.Linear(hid_dim, input_dim)
        self.t_fc1 = nn.Linear(total_input_dim, hid_dim)
        self.t_fc2 = nn.Linear(hid_dim, hid_dim)
        self.t_fc3 = nn.Linear(hid_dim, input_dim)
        self.mask = nn.Parameter(mask, requires_grad=False)
        self.s_tanh_activation = s_tanh_activation
        self.smooth_activation = smooth_activation

    def forward(self, x, cond_x=None, mode='direct'):
        x_m = x * self.mask
        if cond_x is not None:
            x_m = torch.cat([x_m, cond_x.expand(x_m.shape[0], -1, -1)], -1)
        if self.smooth_activation:
            if self.s_tanh_activation:
                s_out = torch.tanh(self.s_fc3(F.elu(self.s_fc2(F.elu(self.s_fc1(x_m)))))) * (1-self.mask)
            else:
                s_out = self.s_fc3(F.elu(self.s_fc2(F.elu(self.s_fc1(x_m))))) * (1-self.mask)
            t_out = self.t_fc3(F.elu(self.t_fc2(F.elu(self.t_fc1(x_m))))) * (1-self.mask)
        else:
            if self.s_tanh_activation:
                s_out = torch.tanh(self.s_fc3(F.relu(self.s_fc2(F.relu(self.s_fc1(x_m)))))) * (1-self.mask)
            else:
                s_out = self.s_fc3(F.relu(self.s_fc2(F.relu(self.s_fc1(x_m))))) * (1-self.mask)
            t_out = self.t_fc3(F.relu(self.t_fc2(F.relu(self.t_fc1(x_m))))) * (1-self.mask)
        if mode == 'direct':
            y = x * torch.exp(s_out) + t_out
            log_det_jacobian = s_out.sum(-1, keepdim=True)
        else:
            y = (x - t_out) * torch.exp(-s_out)
            log_det_jacobian = -s_out.sum(-1, keepdim=True)
        return y, log_det_jacobian

class RealNVP(nn.Module):
    def __init__(self, input_dim, hid_dim = 256, n_layers = 2, cond_dim = None, s_tanh_activation = True, smooth_activation=False):
        super().__init__()
        assert n_layers >= 2, 'num of coupling layers should be greater or equal to 2'
        
        self.input_dim = input_dim
        mask = (torch.arange(0, input_dim) % 2).float()
        self.modules = []
        self.modules.append(CouplingLayer(input_dim, hid_dim, mask, cond_dim, s_tanh_activation, smooth_activation))
        for _ in range(n_layers - 2):
            mask = 1 - mask
            self.modules.append(CouplingLayer(input_dim, hid_dim, mask, cond_dim, s_tanh_activation, smooth_activation))
        self.modules.append(CouplingLayer(input_dim, hid_dim, 1 - mask, cond_dim, s_tanh_activation, smooth_activation))
        self.module_list = nn.ModuleList(self.modules)
        
    def forward(self, x, cond_x=None, mode='direct'):
        """ Performs a forward or backward pass for flow modules.
        Args:
            x: a tuple of inputs and logdets
            mode: to run direct computation or inverse
        """
        logdets = torch.zeros(x.size(), device=x.device).sum(-1, keepdim=True)

        assert mode in ['direct', 'inverse']
        if mode == 'direct':
            for module in self.module_list:
                x, logdet = module(x, cond_x, mode)
                logdets += logdet
        else:
            for module in reversed(self.module_list):
                x, logdet = module(x, cond_x, mode)
                logdets += logdet

        return x, logdets

    def log_probs(self, x, cond_x = None):
        u, log_jacob = self(x, cond_x)
        log_probs = (-0.5 * u.pow(2) - 0.5 * math.log(2 * math.pi)).sum(
            -1, keepdim=True)
        return (log_probs + log_jacob).sum(-1, keepdim=True)

    def sample(self, num_samples, noise=None, cond_x=None):
        if noise is None:
            noise = torch.Tensor(num_samples, self.input_dim).normal_()
        device = next(self.parameters()).device
        noise = noise.to(device)
        if cond_x is not None:
            cond_x = cond_x.to(device)
        samples = self.forward(noise, cond_x, mode='inverse')[0]
        return samples
    
class RealNVPVAE(nn.Module):

    def __init__(self, latent_dim=50):
        super().__init__()
        self.latent_dim = latent_dim
        self.G = GenerativeModel(latent_dim)
        self.F = RealNVP(latent_dim, hid_dim=64, n_layers=6, cond_dim=784)

    def energy_0(self, x, y):
        return (x**2).sum(axis=2, keepdims=True) / 2
    
    def sample_energy_0(self, y, M):
        device = next(self.parameters()).device
        x = torch.randn(M, y.shape[0], self.latent_dim).to(device)
        return x
        
    def energy_1(self, x, y):
        return (x**2).sum(axis=2, keepdims=True) / 2 - self.G.conditional_log_likelihood(x, y).sum(axis=2, keepdims=True)

    def forward(self, x, y):
        dW = self.energy_0(x, y)
        x, tmp_dW = self.F(x, y)
        dW += tmp_dW
        dW = dW - self.energy_1(x, y)
        return x, dW

    def log_likelihood(self, y, M):
        x0 = self.sample_energy_0(y.view(-1, 784), M)
        x, dW = self.forward(x0, y.view(-1, 784))
        return torch.mean(dW, axis=0, keepdims=False)

class RealNVPVAE_eval(nn.Module):

    def __init__(self, G):
        super().__init__()
        latent_dim = G.latent_dim
        self.latent_dim = latent_dim
        self.G = G
        self.F = RealNVP(latent_dim, hid_dim=256, n_layers=12, cond_dim=784)

    def energy_0(self, x, y):
        return (x**2).sum(axis=2, keepdims=True) / 2
    
    def sample_energy_0(self, y, M):
        device = next(self.parameters()).device
        x = torch.randn(M, y.shape[0], self.latent_dim).to(device)
        return x
        
    def energy_1(self, x, y):
        return (x**2).sum(axis=2, keepdims=True) / 2 - self.G.conditional_log_likelihood(x, y).sum(axis=2, keepdims=True)

    def forward(self, x, y):
        dW = self.energy_0(x, y)
        x, tmp_dW = self.F(x, y)
        dW += tmp_dW
        dW = dW - self.energy_1(x, y)
        return x, dW

    def log_likelihood(self, y, M):
        x0 = self.sample_energy_0(y.view(-1, 784), M)
        x, dW = self.forward(x0, y.view(-1, 784))
        return torch.logsumexp(dW, axis=0, keepdims=False) - math.log(M)

def ModelEval(G, sample_size, data_file):
    start = time.process_time()
    
    device = torch.device("cuda")
    latent_dim = 50
    batch_size = 128
    n_epochs = 40
    log_interval = 10

    if data_file == 'mnist_data':
        train_loader = torch.utils.data.DataLoader(
            datasets.MNIST(data_file, train=True, download=False,
                           transform=transforms.ToTensor()),
                            batch_size=batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(
            datasets.MNIST(data_file, train=False, transform=transforms.ToTensor()),
            batch_size=batch_size, shuffle=False)
    else:
        train_loader = torch.utils.data.DataLoader(
            datasets.FashionMNIST(data_file, train=True, download=False,
                           transform=transforms.ToTensor()),
                            batch_size=batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(
            datasets.FashionMNIST(data_file, train=False, transform=transforms.ToTensor()),
            batch_size=batch_size, shuffle=False)
        
    flow = RealNVPVAE_eval(G).to(device)
    optim = torch.optim.Adam(flow.F.parameters(), lr=1e-3)

    M = 1
    for epoch in range(1, n_epochs + 1):
        train_loss = 0
        for batch_idx, (data, _) in enumerate(test_loader):
            data = ((torch.rand_like(data) <= data) + 0.).float()
            data = data.to(device)
            loss = -flow.log_likelihood(data, M).mean()
            optim.zero_grad()
            loss.backward()
            train_loss += loss.item()*len(data)
            optim.step()
            if batch_idx % log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(test_loader.dataset),
                    100. * batch_idx / len(test_loader),
                    loss.item()))
    
    with torch.no_grad():
        test_loss = 0
        M = sample_size
        K = 10
        for kk in range(K):
            for batch_idx, (data, _) in enumerate(test_loader):
                data = ((torch.rand_like(data) <= data) + 0.).float()
                data = data.to(device)
                loss = -flow.log_likelihood(data, M).mean()
                test_loss += loss.item()*len(data)
                if batch_idx % log_interval == 0:
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        kk, batch_idx * len(data), len(test_loader.dataset),
                        100. * batch_idx / len(test_loader),
                        loss.item()))
        test_loss /= len(test_loader.dataset)*K
    print('====> Test set NLL: {:.4f}'.format(test_loss))

    return test_loss
    

class SNFVAE(nn.Module):

    def __init__(self, latent_dim=50, unit_num=3, nsteps=10, stepsize=0.1):
        super().__init__()
        self.latent_dim = latent_dim
        self.unit_num = unit_num
        self.G = GenerativeModel(latent_dim)
        self.F_list = []
        for _ in range(unit_num):
            self.F_list.append(RealNVP(latent_dim, hid_dim=64, n_layers=2, cond_dim=784))
        self.F_list = nn.ModuleList(self.F_list)
        self.nsteps = nsteps
        stepsize_list = torch.FloatTensor([stepsize,] * nsteps * unit_num)
        lambda_list = (np.array(range(1,nsteps * unit_num + 1))/nsteps / unit_num).tolist()
        lambda_list = torch.FloatTensor(lambda_list)
        self.stepsize_para_list, self.lambda_para_list = self.stepsize_lambda_2_para(stepsize_list, lambda_list)
        self.stepsize_para_list = nn.Parameter(torch.FloatTensor(self.stepsize_para_list), requires_grad=True)
        self.lambda_para_list = nn.Parameter(torch.FloatTensor(self.lambda_para_list))
        
    def stepsize_lambda_2_para(self, stepsize_list, lambda_list):
        stepsize_para_list = torch.clamp(torch.abs(stepsize_list), min=1e-6)
        lambda_para_list = lambda_list
        return stepsize_para_list, lambda_para_list
    
    def para_2_stepsize_lambda(self, stepsize_para_list, lambda_para_list):
        stepsize_list = torch.abs(stepsize_para_list) + 1e-6
        lambda_list = lambda_para_list
        return stepsize_list, lambda_list

    def energy_0(self, x, y):
        return (x**2).sum(axis=2, keepdims=True) / 2

    def force_0(self, x, y):
        return -x
    
    def sample_energy_0(self, y, M):
        device = next(self.parameters()).device
        x = torch.randn(M, y.shape[0], self.latent_dim).to(device)
        return x
        
    def energy_1(self, x, y):
        return (x**2).sum(axis=2, keepdims=True) / 2 - self.G.conditional_log_likelihood(x, y).sum(axis=2, keepdims=True)

    def force_1(self, x, y):
        x0 = x.clone().detach().requires_grad_(True)
        e = self.energy_1(x0, y)
        return -torch.autograd.grad(e.sum(), x0, create_graph=True)[0]

    def interpolated_energy(self, x, y, lambda_=1.):
        return self.energy_0(x, y) * (1 - lambda_) + self.energy_1(x, y) * lambda_

    def interpolated_force(self, x, y, lambda_=1.):
        return self.force_0(x, y) * (1 - lambda_) + self.force_1(x, y) * lambda_

    def forward(self, x, y, flow_disable=False):
        stepsize_list, lambda_list = self.para_2_stepsize_lambda(self.stepsize_para_list, self.lambda_para_list)
        dW = self.energy_0(x, y)
        for i in range(self.nsteps * self.unit_num):
            if i % self.nsteps == 0:
                x, tmp_dW = self.F_list[int(i/self.nsteps)](x, y)
                dW += tmp_dW                
            if flow_disable:
                continue
            lambda_ = lambda_list[i]
            stepsize = stepsize_list[i]
            # forward step
            x1 = x + stepsize * self.interpolated_force(x, lambda_) + torch.sqrt(2*stepsize) * torch.randn_like(x)
            tmp_dW = self.interpolated_energy(x1, y, lambda_) - self.interpolated_energy(x, y, lambda_)
            A = torch.exp(torch.clamp(-tmp_dW, - math.inf, 0.))
            u = torch.rand_like(A)
            acc = (u <= A).float()
            x = (1 - acc) * x + acc * x1
            dW += acc * tmp_dW
        dW = dW - self.energy_1(x, y)
        return x, dW

    def log_likelihood(self, y, M, flow_disable=False):
        x0 = self.sample_energy_0(y.view(-1, 784), M)
        x, dW = self.forward(x0, y.view(-1, 784), flow_disable)
        return torch.mean(dW, axis=0, keepdims=False)
