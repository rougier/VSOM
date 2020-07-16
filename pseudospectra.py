import numpy as np
import matplotlib.pylab as plt
from forbiddenfruit import curse

from rectpsa import psa

curse(np.ndarray, 'H', property(fget=lambda A: A.conj().T))


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


def compute_pseudospectra(X, is_plot=True):
    check_normality(X)

    npts = 100
    xmin, xmax = -1.0, 3.5
    ymin, ymax = -1.5, 1.5
    Xx, Yy = np.meshgrid(np.linspace(xmin, xmax, npts),
                         np.linspace(ymin, ymax, npts))
    spectra, s_max = psa(X, Xx, Yy, method='svd')

    fig = plt.figure()
    ax = fig.add_subplot(111)
    circles = np.logspace(np.log10(10e-4), np.log10(1.5), 20)[:-3]
    CS = ax.contour(Xx, Yy, spectra+1e-20, levels=circles)
    ax.clabel(CS, inline=1, fontsize=10)
    return spectrum


def activation(som, samples):
    codebook = som.codebook["X"]
    p, n = codebook.shape[0], samples.shape[0]
    Y = np.zeros((p, n))
    for i, s in enumerate(samples):
        tmp = -np.sqrt(((codebook - s)**2).sum(axis=-1))
        Y[:, i] = tmp
    return Y


def spectrum(X):
    w, v = np.linalg.eig(X)
    return w, v


def ps_on_gram(som, samples):
    X = activation(som, samples)
    G = grammian(X)
    W, V = spectrum(G)
    print(V.shape)
    print(W)
    # plt.figure()
    # plt.scatter()
    compute_pseudospectra(G)
