import numpy as np

def sample_energy(bg, n_samples, n_repeat, nbins=50):
    hist_x = None
    whist_x = None
    hists_y = []
    whists_y = []
    for i in range(n_repeat):
        # sample
        z = bg.prior.sample(n_samples, temperature=1.0)
        y, dW = bg.flow(z, temperature=1.0)
        # 
        samples = y[:, 0]
        pathweights = dW[:, 0]
        energies_0 = bg._prior.energy(z)[:, 0]
        energies_T = bg._target.energy(y)[:, 0]
        log_totalweights = pathweights - energies_T + energies_0
        log_totalweights -= log_totalweights.mean()
        # 
        hist_y, edges = np.histogram(samples.detach().numpy(), bins=nbins, density=True)
        hists_y.append(-np.log(hist_y))
        hist_x = 0.5 * (edges[1:] + edges[:-1])
        #
        whist_y, edges = np.histogram(samples.detach().numpy(), bins=nbins, density=True,
                                      weights=np.exp(log_totalweights.detach().numpy()))
        whists_y.append(-np.log(whist_y))
        whist_x = 0.5 * (edges[1:] + edges[:-1])

    # align energies
    for i in range(n_repeat):
        hists_y[i] -= np.mean(hists_y[i][np.isfinite(hists_y[i])])
        whists_y[i] -= np.mean(whists_y[i][np.isfinite(whists_y[i])])

    return hist_x, hists_y, whist_x, whists_y

def statistical_efficiency(bg, n_samples=20000, n_resample=100):
    z = bg._prior.sample(n_samples, temperature=1.0)
    y, dW = bg.flow(z, temperature=1.0)

    samples = y[:, 0]
    pathweights = dW[:, 0]
    energies_0 = bg._prior.energy(z)[:, 0]
    energies_T = bg._target.energy(y)[:, 0]
    log_totalweights = pathweights - energies_T + energies_0
    log_totalweights = log_totalweights.detach().numpy().astype(np.float64)

    statistical_efficiencies = []
    for i in range(n_resample):
        subsample = np.random.choice(log_totalweights, size=log_totalweights.size)
        subsample -= subsample.max()
        statistical_efficiencies.append(np.exp(subsample).sum() / np.size(subsample))

    return np.mean(statistical_efficiencies), np.std(statistical_efficiencies)/np.sqrt(n_resample)