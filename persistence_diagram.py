import mnist
import numpy as np
import gudhi as gd
import matplotlib
import matplotlib.pylab as plt
from gudhi.representations import vector_methods

matplotlib.use('Agg')


def disc(n_points=20):
    data = np.zeros((n_points, 2))
    r = np.sqrt(np.random.uniform(0.5, 1, size=(n_points, )))
    theta = np.pi * np.random.uniform(0, 2, size=(n_points, ))
    data[:, 0] = r * np.cos(theta)
    data[:, 1] = r * np.sin(theta)
    return data


def dgm(data, is_alpha_simplex_on=False):
    cp = data.tolist()
    if is_alpha_simplex_on:
        skeleton = gd.AlphaComplex(points=cp)
        alpha_simplex_tree = skeleton.create_simplex_tree()
        bar_codes = alpha_simplex_tree.persistence()
        dim0 = alpha_simplex_tree.persistence_intervals_in_dimension(0)
        dim1 = alpha_simplex_tree.persistence_intervals_in_dimension(1)
        dim2 = alpha_simplex_tree.persistence_intervals_in_dimension(2)
    else:
        # skeleton = gd.RipsComplex(points=cp, max_edge_length=1.3)
        skeleton = gd.RipsComplex(points=cp, max_edge_length=12.0)
        rips_simplex_tree = skeleton.create_simplex_tree(max_dimension=2)
        bar_codes = rips_simplex_tree.persistence()
        dim0 = rips_simplex_tree.persistence_intervals_in_dimension(0)
        dim1 = rips_simplex_tree.persistence_intervals_in_dimension(1)
        dim2 = rips_simplex_tree.persistence_intervals_in_dimension(2)
    return bar_codes, dim0, dim1, dim2


def run_pd(dataX, dataY, dataZ, case='Persistence Homology'):
    bcX, iX0, iX1, _ = dgm(dataX, is_alpha_simplex_on=False)
    bcY, iY0, iY1, _ = dgm(dataY, is_alpha_simplex_on=False)
    bcZ, iZ0, iZ1, _ = dgm(dataZ, is_alpha_simplex_on=False)

    print(30*"*")
    entropy = vector_methods.Entropy()
    print("Input Entropy DH0", entropy(np.nan_to_num(iX0)))
    print("Input Entropy DH1", entropy(np.nan_to_num(iX1)))
    print("Regular Entropy DH0", entropy(np.nan_to_num(iY0)))
    print("Regular Entropy DH1", entropy(np.nan_to_num(iY1)))
    print("Random Entropy DH0", entropy(np.nan_to_num(iZ0)))
    print("Random Entropy DH1", entropy(np.nan_to_num(iZ1)))

    print("Regular Bottleneck DH0: ", gd.bottleneck_distance(iX0, iY0))
    print("Random Bottleneck DH0: ", gd.bottleneck_distance(iX0, iZ0))

    print("Regular Bottleneck DH1: ", gd.bottleneck_distance(iX1, iY1))
    print("Random Bottleneck DH1: ", gd.bottleneck_distance(iX1, iZ1))

    print("Regular Wasserstein DH0: ", gd.hera.wasserstein_distance(iX0, iY0))
    print("Random Wasserstein DH0: ", gd.hera.wasserstein_distance(iX0, iZ0))

    print("Regular Wasserstein DH1: ", gd.hera.wasserstein_distance(iX1, iY1))
    print("Random Wasserstein DH1: ", gd.hera.wasserstein_distance(iX1, iZ1))
    print(30*"*")


    fig = plt.figure(figsize=(16, 11))
    fig.suptitle(case, fontsize=18, weight='bold')
    ax1 = fig.add_subplot(231)
    gd.plot_persistence_barcode(bcX, axes=ax1)
    ax1.set_title("")
    # ax1.set_xlim(0, .6)
    ax1.set_xlabel(r"$\alpha$", fontsize=21, weight='bold')

    ax2 = fig.add_subplot(232)
    gd.plot_persistence_barcode(bcY, axes=ax2)
    ax2.set_title("")
    # ax2.set_xlim(0, .6)
    ax2.set_xlabel(r"$\alpha$", fontsize=21, weight='bold')

    ax3 = fig.add_subplot(233)
    gd.plot_persistence_barcode(bcZ, axes=ax3)
    ax3.set_title("")
    # ax3.set_xlim(0, .6)
    ax3.set_xlabel(r"$\alpha$", fontsize=21, weight='bold')

    ax4 = fig.add_subplot(234)
    gd.plot_persistence_diagram(bcX, axes=ax4, greyblock=False, legend=True,
                                fontsize=21)
    ax4.set_title("")
    ticks = np.round(ax4.get_xticks(), 2)
    ax4.set_xticklabels(ticks, fontsize=18, weight='bold')

    ax5 = fig.add_subplot(235)
    gd.plot_persistence_diagram(bcY, axes=ax5, greyblock=False, legend=True,
                                fontsize=21)
    ticks = np.round(ax5.get_xticks(), 2)
    ax5.set_xticklabels(ticks, fontsize=18, weight='bold')
    ax5.set_ylabel("")
    ax5.set_title("")

    ax6 = fig.add_subplot(236)
    gd.plot_persistence_diagram(bcZ, axes=ax6, greyblock=False, legend=True,
                                fontsize=21)
    ticks = np.round(ax6.get_xticks(), 2)
    ax6.set_xticklabels(ticks, fontsize=18, weight='bold')
    ax6.set_title("")
    ax6.set_ylabel("")

    return iX0, iX1, iY0, iY1, iZ0, iZ1


def run_betti(iX0, iX1, iY0, iY1):
    betti_seq = vector_methods.BettiCurve(resolution=100,
                                          sample_range=[np.nan, np.nan])
    landscape = vector_methods.Landscape(num_landscapes=5, resolution=100,
                                         sample_range=[np.nan, np.nan])

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(221)
    ax.plot(betti_seq(iX0[:-1]), 'k', alpha=0.6, label="Regular B0")
    ax.plot(betti_seq(iY0[:-1]), 'b', alpha=0.6, label="Random B0")
    ax.legend()

    ax = fig.add_subplot(222)
    ax.plot(betti_seq(iX1[:-1]), 'k', alpha=0.6, label="Regular B1")
    ax.plot(betti_seq(iY1[:-1]), 'b', alpha=0.6, label="Random B1")
    ax.legend()

    ax = fig.add_subplot(223)
    ax.plot(landscape(iX0[:-1]), 'k', alpha=0.6, label="Regular Landscape 0")
    ax.plot(landscape(iY0[:-1]), 'b', alpha=0.6, label="Random Landscape 0")
    ax.legend()

    ax = fig.add_subplot(224)
    ax.plot(landscape(iX1[:-1]), 'k', alpha=0.6, label="Regular Landscape 1")
    ax.plot(landscape(iY1[:-1]), 'b', alpha=0.6, label="Random Landscape 1")
    ax.legend()


if __name__ == '__main__':
    base = "Experiment "
    cases = ["2 - Annulus", "2b - Holes", "3 - Uniform Cube", "4 - MNIST"]
    # np.random.seed(1)
    # dataY = np.load("./data/experiment-2-regular.npy")
    # dataZ = np.load("./data/experiment-2-random.npy")
    # n_samples = dataY.shape[0]
    # dataX = np.random.uniform(0, 1, (n_samples, 2))
    # run_pd(dataX, dataY, dataZ, case=base+cases[0])
    # plt.savefig("./figures/experiment-2-pd.pdf", axis='tight')

    # np.random.seed(12345)
    # dataY = np.load("./data/experiment-2-bis-regular.npy")
    # dataZ = np.load("./data/experiment-2-bis-random.npy")
    # n_samples = dataY.shape[0]
    # X = np.random.uniform(0, 1, n_samples)
    # Y = np.random.uniform(0, 1, n_samples)
    # holes = 64
    # for i in range(holes):
    #     x, y = np.random.uniform(0.1, 0.9, 2)
    #     r = 0.1 * np.random.uniform(0, 1)
    #     ind = ((X-x)**2 + (Y-y)**2) > r*r
    #     X, Y = X[ind], Y[ind]
    # dataX = np.c_[X, Y]
    # run_pd(dataX, dataY, dataZ, case=base+cases[1])
    # plt.savefig("./figures/experiment-2-bis-pd.pdf", axis='tight')

    # np.random.seed(1)
    # dataY = np.load("./data/experiment-3-regular.npy")
    # dataZ = np.load("./data/experiment-3-random.npy")
    # n_samples = dataY.shape[0]
    # dataX = np.random.uniform(0, 1, (n_samples, 3))
    # run_pd(dataX, dataY, dataZ, case=base+cases[2])
    # plt.savefig("./figures/experiment-3-pd.pdf", axis='tight')

    np.random.seed(1)
    dataY = np.load("./data/experiment-4-regular.npy")
    dataZ = np.load("./data/experiment-4-random.npy")
    dataX, L = mnist.read("training")
    N = 1000
    index = np.random.choice(np.arange(0, 50000), N)
    dataX = dataX[index].reshape(N, 28*28)
    # dataX[dataX != 0] = 1
    run_pd(dataX, dataY, dataZ, case=base+cases[3])
    plt.savefig("./figures/experiment-4-pd.pdf", axis='tight')
    # plt.show()
