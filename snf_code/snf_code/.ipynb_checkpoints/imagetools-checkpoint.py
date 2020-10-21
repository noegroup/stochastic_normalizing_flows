import torch
import numpy as np

from bgtorch import BoltzmannGenerator
from bgtorch.nn.flow import SequentialFlow, SplitFlow, MergeFlow, SwapFlow
from bgtorch.nn.flow.stochastic import MetropolisMCFlow
from snf_code import RNVP, NSF, InterpolatedEnergy


def boltzmann_generator_NSF_MC(prior, target, n_transform, 
                               n_bins=20,
                               tail=1,
                               width_nhidden=[64, 64, 64], 
                               height_nhidden=[64, 64, 64],
                               slope_nhidden=[64, 64, 64],
                               stochastic=False, diffuse_at_0=False, nrelax=20, stepsize=0.1):
    # here we aggregate all layers of the flow
    layers = []

    # first flow
    if diffuse_at_0:
        layers.append(MetropolisMCFlow(prior, nsteps=nrelax, stepsize=stepsize))

    # split
    layers.append(SplitFlow(prior.dim//2))
        
    # RealNVP block
    for i in range(n_transform):
        # ic(x) -> v
        layers.append(NSF([prior.dim//2] + width_nhidden + [n_bins * prior.dim//2], 
                          [prior.dim//2] + height_nhidden + [n_bins * prior.dim//2], 
                          [prior.dim//2] + slope_nhidden + [(n_bins + 1) * prior.dim//2], 
                          width_activation=torch.nn.ReLU(),
                          height_activation=torch.nn.ReLU(),
                          slope_activation=torch.nn.ReLU(),
                          n_bins=n_bins,
                          tail=tail
                         ))
        layers.append(SwapFlow())
                            
        # v -> ic(x)
        layers.append(NSF([prior.dim//2] + width_nhidden + [n_bins * prior.dim//2], 
                          [prior.dim//2] + height_nhidden + [n_bins * prior.dim//2], 
                          [prior.dim//2] + slope_nhidden + [(n_bins + 1) * prior.dim//2], 
                          width_activation=torch.nn.ReLU(),
                          height_activation=torch.nn.ReLU(),
                          slope_activation=torch.nn.ReLU(),
                          n_bins=n_bins,
                          tail=tail
                         ))
        layers.append(SwapFlow())

        if stochastic and i < n_transform-1:
            layers.append(MergeFlow(prior.dim//2))
            
            lambda_ = i * 1.0/(n_transform-1)
            energy_model = InterpolatedEnergy(prior, target, lambda_)
            layers.append(MetropolisMCFlow(energy_model, nsteps=nrelax, stepsize=stepsize))
        
            layers.append(SplitFlow(prior.dim//2))

    # merge
    layers.append(MergeFlow(prior.dim//2))
    
    # final flow
    if stochastic:
        layers.append(MetropolisMCFlow(target, nsteps=nrelax, stepsize=stepsize))

    # now define the flow as a sequence of all operations stored in layers
    flexflow = SequentialFlow(layers)
    
    bg = BoltzmannGenerator(prior, flexflow, target)
    
    return bg



def boltzmann_generator_RNVP_MC(prior, target, n_transform, shift_nhidden=[64, 64, 64], scale_nhidden=[64, 64, 64],
                                stochastic=False, diffuse_at_0=False, nrelax=20, stepsize=0.1):
    # here we aggregate all layers of the flow
    layers = []

    # first flow
    if diffuse_at_0:
        layers.append(MetropolisMCFlow(prior, nsteps=nrelax, stepsize=stepsize))

    # split
    layers.append(SplitFlow(prior.dim//2))
        
    # RealNVP block
    for i in range(n_transform):
        # ic(x) -> v
        layers.append(RNVP([prior.dim//2] + shift_nhidden + [prior.dim//2], 
                           [prior.dim//2] + scale_nhidden + [prior.dim//2], 
                           shift_activation=torch.nn.ReLU(), scale_activation=torch.nn.ReLU()))
        layers.append(SwapFlow())
                            
        # v -> ic(x)
        layers.append(RNVP([prior.dim//2] + shift_nhidden + [prior.dim//2], 
                           [prior.dim//2] + scale_nhidden + [prior.dim//2], 
                           shift_activation=torch.nn.ReLU(), scale_activation=torch.nn.ReLU()))
        layers.append(SwapFlow())

        if stochastic and i < n_transform-1:
            layers.append(MergeFlow(prior.dim//2))
            
            lambda_ = i * 1.0/(n_transform-1)
            energy_model = InterpolatedEnergy(prior, target, lambda_)
            layers.append(MetropolisMCFlow(energy_model, nsteps=nrelax, stepsize=stepsize))
        
            layers.append(SplitFlow(prior.dim//2))

    # merge
    layers.append(MergeFlow(prior.dim//2))
    
    # final flow
    if stochastic:
        layers.append(MetropolisMCFlow(target, nsteps=nrelax, stepsize=stepsize))

    # now define the flow as a sequence of all operations stored in layers
    flexflow = SequentialFlow(layers)
    
    bg = BoltzmannGenerator(prior, flexflow, target)
    
    return bg


def sample_bg_histogram(bg, nsamples=100000, nbins=50, xrange=(-1, 1), yrange=(-1, 1)):
    # sample
    Z_ = bg.prior.sample(nsamples)
    Y_, _ = bg.flow(Z_)
    Y_ = Y_.detach().numpy()

    hist_Y_, _, _ = np.histogram2d(Y_[:, 0], Y_[:, 1], bins=nbins, range=[xrange, yrange])
    hist_Y_ /= hist_Y_.sum()
    return hist_Y_


def kldiv(X, Y, reg_X=1e-10, reg_Y=1e-10):
    Xnorm = X / X.sum()
    Xreg = X + reg_X
    Xreg /= Xreg.sum()
    Yreg = Y + reg_Y
    Yreg /= Yreg.sum()
    s1 = (Xnorm * np.log(Xreg / Yreg)).sum()
    return s1