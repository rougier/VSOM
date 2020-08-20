import numpy as np
import gudhi as gd
from gudhi.representations import vector_methods
import matplotlib.pylab as plt

# from sklearn.cluster import SpectralClustering


def disc(n_points=20):
    data = np.zeros((n_points, 2))
    r = np.sqrt(np.random.uniform(0.5, 1, size=(n_points, )))
    theta = np.pi * np.random.uniform(0, 2, size=(n_points, ))
    data[:, 0] = r * np.cos(theta)
    data[:, 1] = r * np.sin(theta)
    return data


def dgm(data):
    cp = data.tolist()
    # skeleton = gd.AlphaComplex(points=cp)
    # alpha_simplex_tree = skeleton.create_simplex_tree()
    # bar_codes = alpha_simplex_tree.persistence()
    # dim0 = alpha_simplex_tree.persistence_intervals_in_dimension(0)
    # dim1 = alpha_simplex_tree.persistence_intervals_in_dimension(1)
    # dim2 = alpha_simplex_tree.persistence_intervals_in_dimension(2)
    skeleton = gd.RipsComplex(points=cp, max_edge_length=1.3)
    rips_simplex_tree = skeleton.create_simplex_tree(max_dimension=2)
    bar_codes = rips_simplex_tree.persistence()
    dim0 = rips_simplex_tree.persistence_intervals_in_dimension(0)
    dim1 = rips_simplex_tree.persistence_intervals_in_dimension(1)
    dim2 = rips_simplex_tree.persistence_intervals_in_dimension(2)

    fig = plt.figure(figsize=(13, 7))
    ax = fig.add_subplot(121)
    gd.plot_persistence_barcode(bar_codes, axes=ax)
    ax.set_xlim([0, 2])
    ax = fig.add_subplot(122)
    gd.plot_persistence_diagram(bar_codes, axes=ax, greyblock=False)
    return bar_codes, dim0, dim1, dim2


if __name__ == '__main__':
    # cp = gs.circle(10)
    cp = np.load("./data/experiment-2-bis-regular.npy")[:512, :]
    bc0, i00, i01, _ = dgm(cp)

    # cp = np.array([[.2, .2], [.4, .2], [.2, .3], [1.7, 2], [1, .9], [.8, .8],
    #                [.7, .9], [1.1, .9]])
    cp = np.load("./data/experiment-2-bis-random.npy")[:512, :]
    bc1, i10, i11, _ = dgm(cp)

    print(gd.bottleneck_distance(i00, i10))
    print(gd.bottleneck_distance(i01, i11))

    betti_seq = vector_methods.BettiCurve(resolution=100,
                                          sample_range=[np.nan, np.nan])
    landscape = vector_methods.Landscape(num_landscapes=5, resolution=100,
                                         sample_range=[np.nan, np.nan])
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(221)
    ax.plot(betti_seq(i00[:-1]), 'k', alpha=0.6, label="Regular B0")
    ax.plot(betti_seq(i10[:-1]), 'b', alpha=0.6, label="Random B0")
    ax.legend()

    ax = fig.add_subplot(222)
    ax.plot(betti_seq(i01[:-1]), 'k', alpha=0.6, label="Regular B1")
    ax.plot(betti_seq(i11[:-1]), 'b', alpha=0.6, label="Random B1")
    ax.legend()

    ax = fig.add_subplot(223)
    ax.plot(landscape(i00[:-1]), 'k', alpha=0.6, label="Regular Landscape 0")
    ax.plot(landscape(i10[:-1]), 'b', alpha=0.6, label="Random Landscape 0")
    ax.legend()

    ax = fig.add_subplot(224)
    ax.plot(landscape(i01[:-1]), 'k', alpha=0.6, label="Regular Landscape 1")
    ax.plot(landscape(i11[:-1]), 'b', alpha=0.6, label="Random Landscape 1")
    ax.legend()
    plt.savefig("experiment-2d-bis-betty.pdf")
    # clustering = SpectralClustering(n_clusters=2,
    #                                 assign_labels='discretize').fit(cp)
    # print(clustering.labels_)
    plt.show()
