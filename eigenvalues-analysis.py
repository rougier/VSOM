import mnist
import numpy as np
import matplotlib.pylab as plt
from forbiddenfruit import curse
from sklearn.neighbors import KernelDensity
from scipy.stats import wasserstein_distance

curse(np.ndarray, 'H', property(fget=lambda A: A.conj().T))

CB_color_cycle = ['#377eb8', '#ff7f00', '#4daf4a',
                  '#f781bf', '#a65628', '#984ea3',
                  '#999999', '#e41a1c', '#dede00']


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


def activation(codebook, samples):
    p, n = codebook.shape[0], samples.shape[0]
    Y = np.zeros((p, n))
    for i, s in enumerate(samples):
        tmp = -np.sqrt(((codebook - s)**2).sum(axis=-1))
        Y[:, i] = tmp
    return Y


def make_list_equal_size(in_list):
    lens = [len(i) for i in in_list]
    max_len = max(lens)
    out_array = np.zeros((len(lens), max_len))
    for i, tmp_list in enumerate(in_list):
        out_array[i, :lens[i]] = tmp_list
    return out_array


def eigvals_distribution(regular_codebook, random_codebook, ax, case='1d'):
    nsamples, ensemble_size = 200, 100
    regular = []
    random = []
    mnistX, _ = mnist.read("training")
    mnistX = mnistX.reshape(-1, 28*28)
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
        elif case == '4d':
            idx = np.arange(mnistX.shape[0])
            idx = np.random.choice(idx, nsamples)
            samples = mnistX[idx]
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
    if case == '2dbis':
        regular = make_list_equal_size(regular)
        random = make_list_equal_size(random)
    mineig = min(wreg.min(), wran.min())
    maxeig = max(wreg.max(), wran.max())
    regular = np.array(regular).flatten().reshape(-1, 1)
    random = np.array(random).flatten().reshape(-1, 1)
    kde_re = KernelDensity(kernel='gaussian', bandwidth=.1).fit(regular)
    kde_ra = KernelDensity(kernel='gaussian', bandwidth=.1).fit(random)
    # X = np.linspace(-50, 30000, 1000)
    # X = np.linspace(0, 10, 100)
    X = np.linspace(mineig, maxeig, 100)
    X = X[:, np.newaxis]
    P = np.exp(kde_re.score_samples(X))
    Q = np.exp(kde_ra.score_samples(X))
    ax.plot(P, 'b', lw=2, label='SOM (P)', c=CB_color_cycle[0], ls='--')
    ax.plot(Q, 'k', lw=2, label='RSOM (Q)', c=CB_color_cycle[1], ls='-.')
    w = np.round(wasserstein_distance(P, Q), 7)
    print(w)
    # ax.set_title(r"$W(P, Q) = $"+str(w), fontsize=16, weight='bold')
    ax.legend()


def eigs():
    fig = plt.figure(figsize=(17, 4))
    fig.subplots_adjust(wspace=0.4, hspace=0.4, bottom=0.2)

    ax = fig.add_subplot(141)
    som_regular = np.load("./data/experiment-2-regular.npy")
    som_random = np.load("./data/experiment-2-random.npy")
    eigvals_distribution(som_regular, som_random, ax, case='2d')
    ax.set_xlim([0, 10])
    ax.set_ylim([0, 3.5])
    ticks = ax.get_xticks().astype('i')
    ax.set_xticklabels(ticks, fontsize=16, weight='bold')
    ticks = np.round(ax.get_yticks(), 4)
    ax.set_yticklabels(ticks, fontsize=16, weight='bold')
    ax.set_xlabel('Eigenvalues', fontsize=16, weight='bold')
    ax.set_ylabel('Probability Density', fontsize=16, weight='bold')
    ax.text(0, 3.8, 'A',
            va='top',
            ha='left',
            fontsize=16,
            weight='bold')

    ax = fig.add_subplot(142)
    som_regular = np.load("./data/experiment-2-bis-regular.npy")
    som_random = np.load("./data/experiment-2-bis-random.npy")
    eigvals_distribution(som_regular, som_random, ax, case='2dbis')
    ax.set_xlim([0, 10])
    ax.set_ylim([0, 3.5])
    ax.set_xlabel('Eigenvalues', fontsize=16, weight='bold')
    ticks = ax.get_xticks().astype('i')
    ax.set_xticklabels(ticks, fontsize=16, weight='bold')
    ax.set_yticks([])
    # ticks = np.round(ax.get_yticks(), 4)
    # ax.set_yticklabels(ticks, fontsize=16, weight='bold')
    ax.text(0, 3.8, 'B',
            va='top',
            ha='left',
            fontsize=16,
            weight='bold')

    ax = fig.add_subplot(143)
    som_regular = np.load("./data/experiment-3-regular.npy")
    som_random = np.load("./data/experiment-3-random.npy")
    eigvals_distribution(som_regular, som_random, ax, case='3d')
    ax.set_xlabel('Eigenvalues', fontsize=16, weight='bold')
    ax.set_xlim([0, 10])
    ax.set_ylim([0, 3.5])
    ticks = ax.get_xticks().astype('i')
    ax.set_xticklabels(ticks, fontsize=16, weight='bold')
    ax.set_yticks([])
    # ticks = np.round(ax.get_yticks(), 4)
    # ax.set_yticklabels(ticks, fontsize=16, weight='bold')
    ax.text(0, 3.8, 'C',
            va='top',
            ha='left',
            fontsize=16,
            weight='bold')

    ax = fig.add_subplot(144)
    som_regular = np.load("./data/experiment-4-regular.npy")
    som_random = np.load("./data/experiment-4-random.npy")
    eigvals_distribution(som_regular, som_random, ax, case='4d')
    ax.set_xlabel('Eigenvalues', fontsize=16, weight='bold')
    ax.set_xlim([0, 10])
    ax.set_ylim([0, 3.5])
    ticks = ax.get_xticks().astype('i')
    ax.set_xticklabels(ticks, fontsize=16, weight='bold')
    ax.set_yticks([])
    # ticks = np.round(ax.get_yticks(), 4)
    # ax.set_yticklabels(ticks, fontsize=16, weight='bold')
    ax.text(0, 3.8, 'D',
            va='top',
            ha='left',
            fontsize=16,
            weight='bold')
    plt.savefig("./eig-distributions-new.pdf", axis='tight')


if __name__ == '__main__':
    eigs()
    plt.show()
