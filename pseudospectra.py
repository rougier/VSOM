import numpy as np
import matplotlib
import matplotlib.pylab as plt
from forbiddenfruit import curse

from rectpsa import psa

curse(np.ndarray, 'H', property(fget=lambda A: A.conj().T))

matplotlib.use('Agg')


def check_normality(X):
    res1 = X @ X.H
    res2 = X.H @ X
    if (res1 == res2).all():
        print("Matrix is Normal (AA* = A*A)!")
        return True
    else:
        print("Matrix is not Normal!")
        return False


def grammian(X):
    return X.T @ X


def compute_pseudospectra(X, ax):
    check_normality(X)

    npts = 100
    xmin, xmax = -1.0, 20.5
    ymin, ymax = -1.5, 1.5
    Xx, Yy = np.meshgrid(np.linspace(xmin, xmax, npts),
                         np.linspace(ymin, ymax, npts))
    spectra, s_max = psa(X, Xx, Yy, method='svd')

    circles = np.logspace(np.log10(10e-4), np.log10(1.5), 20)[:-3]
    CS = ax.contour(Xx, Yy, spectra+1e-20, levels=circles)
    ax.clabel(CS, inline=1, fontsize=10)
    return spectra


def spectrum(X):
    w, v = np.linalg.eig(X)
    return w, v


def activation(codebook, samples):
    p, n = codebook.shape[0], samples.shape[0]
    Y = np.zeros((p, n))
    for i, s in enumerate(samples):
        tmp = -np.sqrt(((codebook - s)**2).sum(axis=-1))
        Y[:, i] = tmp
    return Y


def activation_(codebook, samples):
    p, n, m = codebook.shape[0], codebook.shape[1], samples.shape[0]
    Y = np.zeros((m, p))
    for i, s in enumerate(samples):
        tmp = -np.sqrt(((codebook - s)**2).sum(axis=-1))
        Y[i, :] = tmp
    return Y


def ps_on_gram(som, samples, ax):
    X = activation(som, samples)
    G = grammian(X)
    W, V = spectrum(G)
    return compute_pseudospectra(G, ax)


def gram_psa():
    n_samples = 50
    samples = np.linspace(0, 1, n_samples)
    fig = plt.figure(figsize=(28, 10))
    ax = fig.add_subplot(121)
    som = np.load("./data/experiment-1-regular.npy")
    ps_on_gram(som, samples, ax)
    ax = fig.add_subplot(122)
    som = np.load("./data/experiment-1-random.npy")
    ps_on_gram(som, samples, ax)
    plt.savefig("experiment-1-gram.pdf")

    T = np.random.uniform(0.0, 2.0*np.pi, n_samples)
    R = np.sqrt(np.random.uniform(0.50**2, 1.0**2, n_samples))
    samples = 0.5 + np.c_[R*np.cos(T), R*np.sin(T)]/2
    fig = plt.figure(figsize=(13, 7))
    ax = fig.add_subplot(121)
    som = np.load("./data/experiment-2-regular.npy")
    ps_on_gram(som, samples, ax)
    ax = fig.add_subplot(122)
    som = np.load("./data/experiment-2-random.npy")
    ps_on_gram(som, samples, ax)
    # plt.savefig("experiment-2-gram.pdf")

    samples = np.random.uniform(0, 1, (n_samples, 3))
    fig = plt.figure(figsize=(13, 7))
    ax = fig.add_subplot(121)
    som = np.load("./data/experiment-3-regular.npy")
    ps_on_gram(som, samples, ax)
    ax = fig.add_subplot(122)
    som = np.load("./data/experiment-3-random.npy")
    ps_on_gram(som, samples, ax)
    # plt.savefig("experiment-3-gram.pdf")


def eigvals_distribution(regular_codebook, random_codebook, ax, case='1d'):
    nsamples = 100
    level_regular = np.zeros((nsamples))
    level_random = np.zeros((nsamples))
    for i in range(nsamples):
        if case == '1d':
            samples = np.random.uniform(0, 1, (50, ))
        elif case == '2d':
            T = np.random.uniform(0.0, 2.0*np.pi, 50)
            R = np.sqrt(np.random.uniform(0.50**2, 1.0**2, 50))
            samples = 0.5 + np.c_[R*np.cos(T), R*np.sin(T)]/2
        else:
            samples = np.random.uniform(0, 1, (50, 3))
        Xreg = activation(regular_codebook, samples)
        Xran = activation(random_codebook, samples)
        Greg = grammian(Xreg)
        Gran = grammian(Xran)
        wreg = np.linalg.eigvalsh(Greg)
        wran = np.linalg.eigvalsh(Gran)
        level_regular[i] = (wreg[1] - wreg[0])
        level_random[i] = (wran[1] - wran[0])
    level_regular /= level_regular.mean()
    level_random /= level_random.mean()
    ax.hist(level_regular, bins=10, color='blue', alpha=0.6, label="Regular")
    ax.hist(level_random, bins=10, color='black', alpha=0.6, label="Random")
    ax.legend()


def eigs():
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    som_regular = np.load("./data/experiment-1-regular.npy")
    som_random = np.load("./data/experiment-1-random.npy")
    eigvals_distribution(som_regular, som_random, ax, case='1d')
    plt.savefig("experiment-1-eigs.pdf")

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    som_regular = np.load("./data/experiment-2-regular.npy")
    som_random = np.load("./data/experiment-2-random.npy")
    eigvals_distribution(som_regular, som_random, ax, case='2d')
    plt.savefig("experiment-2-eigs.pdf")

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    som_regular = np.load("./data/experiment-3-regular.npy")
    som_random = np.load("./data/experiment-3-random.npy")
    eigvals_distribution(som_regular, som_random, ax, case='3d')
    plt.savefig("experiment-3-eigs.pdf")


if __name__ == '__main__':
    # gram_psa()
    eigs()
    # plt.show()
