import numpy as np
import matplotlib
import matplotlib.pylab as plt
from forbiddenfruit import curse
from sklearn.neighbors import KernelDensity
from scipy.stats import wasserstein_distance

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
    fig = plt.figure(figsize=(13, 7))
    ax = fig.add_subplot(121)
    som = np.load("./data/experiment-1-regular.npy")
    ps_on_gram(som, samples, ax)
    ax = fig.add_subplot(122)
    som = np.load("./data/experiment-1-random.npy")
    ps_on_gram(som, samples, ax)
    plt.savefig("./data/experiment-1-psa.pdf")

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
    plt.savefig("./data/experiment-2-psa.pdf")

    n = 50_000
    X = np.random.uniform(0, 1, n)
    Y = np.random.uniform(0, 1, n)
    holes = 64
    for i in range(holes):
        x, y = np.random.uniform(0.1, 0.9, 2)
        r = 0.1 * np.random.uniform(0, 1)
        idx = ((X-x)**2 + (Y-y)**2) > r*r
        X, Y = X[idx], Y[idx]
    X = np.c_[X, Y]
    fig = plt.figure(figsize=(13, 7))
    ax = fig.add_subplot(121)
    som = np.load("./data/experiment-2-bis-regular.npy")
    ps_on_gram(som, samples, ax)
    ax = fig.add_subplot(122)
    som = np.load("./data/experiment-2-bis-random.npy")
    ps_on_gram(som, samples, ax)
    plt.savefig("./data/experiment-2-bis-psa.pdf")

    samples = np.random.uniform(0, 1, (n_samples, 3))
    fig = plt.figure(figsize=(13, 7))
    ax = fig.add_subplot(121)
    som = np.load("./data/experiment-3-regular.npy")
    ps_on_gram(som, samples, ax)
    ax = fig.add_subplot(122)
    som = np.load("./data/experiment-3-random.npy")
    ps_on_gram(som, samples, ax)
    plt.savefig("./data/experiment-3-psa.pdf")


def eigvals_distribution(regular_codebook, random_codebook, ax, case='1d'):
    nsamples, ensemble_size = 100, 500
    regular = []
    random = []
    for i in range(ensemble_size):
        if case == '1d':
            samples = np.random.uniform(0, 1, (nsamples, ))
        elif case == '2d':
            T = np.random.uniform(0.0, 2.0*np.pi, nsamples)
            R = np.sqrt(np.random.uniform(0.50**2, 1.0**2, nsamples))
            samples = 0.5 + np.c_[R*np.cos(T), R*np.sin(T)]/2
        elif case == '2dbis':
            X = np.random.uniform(0, 1, nsamples)
            Y = np.random.uniform(0, 1, nsamples)
            holes = 64
            for i in range(holes):
                x, y = np.random.uniform(0.1, 0.9, 2)
                r = 0.1 * np.random.uniform(0, 1)
                Inp = ((X-x)**2 + (Y-y)**2) > r*r
                X, Y = X[Inp], Y[Inp]
            samples = np.c_[X, Y]
        else:
            samples = np.random.uniform(0, 1, (nsamples, 3))
        Xreg = activation(regular_codebook, samples)
        Xran = activation(random_codebook, samples)
        Greg = grammian(Xreg)
        Gran = grammian(Xran)
        wreg, _ = np.linalg.eig(Greg)
        wran, _ = np.linalg.eig(Gran)
        regular.append(wreg.real)
        random.append(wran.real)
    regular = np.array(regular).flatten().reshape(-1, 1)
    random = np.array(random).flatten().reshape(-1, 1)
    kde_re = KernelDensity(kernel='gaussian', bandwidth=.2).fit(regular)
    kde_ra = KernelDensity(kernel='gaussian', bandwidth=.2).fit(random)
    X = np.linspace(-50, 30000, 1000)
    X = X[:, np.newaxis]
    if case == '2dbis':
        # ax.hist(regular, bins=30, alpha=0.6, label="Regular",
        #         density=True)
        # ax.hist(random, bins=30, alpha=0.6, label="Random",
        #         density=True)
        ax.plot(np.exp(kde_re.score_samples(X)), 'b', lw=2, label='Regular')
        ax.plot(np.exp(kde_ra.score_samples(X)), 'k', lw=2, label='Random')
        ax.legend()
    else:
        P = np.exp(kde_re.score_samples(X))
        Q = np.exp(kde_ra.score_samples(X))
        ax.plot(P, 'b', lw=2, label='Regular')
        ax.plot(Q, 'k', lw=2, label='Random')
        w = wasserstein_distance(P, Q)
        ax.set_title(str(w))
        # ax.hist(regular, bins=30, color='blue', alpha=0.6, label="Regular",
        #         density=True)
        # ax.hist(random, bins=30, color='black', alpha=0.6, label="Random",
        #         density=True)
        ax.legend()


def eigs():
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    som_regular = np.load("./data/experiment-1-regular.npy")
    som_random = np.load("./data/experiment-1-random.npy")
    eigvals_distribution(som_regular, som_random, ax, case='1d')
    plt.savefig("./data/experiment-1-eigs.pdf")

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    som_regular = np.load("./data/experiment-2-regular.npy")
    som_random = np.load("./data/experiment-2-random.npy")
    eigvals_distribution(som_regular, som_random, ax, case='2d')
    plt.savefig("./data/experiment-2-eigs.pdf")

    # fig = plt.figure(figsize=(10, 10))
    # ax = fig.add_subplot(111)
    # som_regular = np.load("./data/experiment-2-bis-regular.npy")
    # som_random = np.load("./data/experiment-2-bis-random.npy")
    # eigvals_distribution(som_regular, som_random, ax, case='2dbis')
    # plt.savefig("./data/experiment-2-bis-eigs.pdf")

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    som_regular = np.load("./data/experiment-3-regular.npy")
    som_random = np.load("./data/experiment-3-random.npy")
    eigvals_distribution(som_regular, som_random, ax, case='3d')
    plt.savefig("./data/experiment-3-eigs.pdf")


if __name__ == '__main__':
    # gram_psa()
    eigs()
    # plt.show()
