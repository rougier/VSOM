import mnist
import numpy as np
import matplotlib.pylab as plt

import umap

from persistence_diagram import persistence
from plot_persistent_homology import plot_diagrams


if __name__ == '__main__':
    np.random.seed(1)
    num_neurons = 1024
    bc_base = './results/barcode-experiment-MNIST-'
    h0_base = './results/homology0-experiment-MNIST-'
    h1_base = './results/homology1-experiment-MNIST-'
    h2_base = './results/homology2-experiment-MNIST-'

    input_space, labels = mnist.read("training")
    index = np.random.choice(np.arange(0, 50000), num_neurons)
    input_space = input_space[index].reshape(num_neurons, 28*28)
    regular_som = np.load("results/experiment-MNIST-regular.npy")
    random_som = np.load("results/experiment-MNIST-random.npy")

    mapper = umap.UMAP(n_components=7).fit(input_space)
    input_mapper = mapper.transform(input_space)
    regular_mapper = mapper.transform(regular_som)
    random_mapper = mapper.transform(random_som)

    data = [input_mapper, regular_mapper, random_mapper]
    # data = [input_space, regular_som, random_som]
    case = ['input_space', 'regular', 'random']

    per = persistence(dimension=3, max_edge_length=1, max_alpha_square=4,
                      is_alpha_simplex_on=True)
    for i, d in enumerate(data):
        per.compute_persistence(d, case='MNIST-'+case[i])

    homology0_input = per.read_pdgm(h0_base+'input_space.dat')
    homology1_input = per.read_pdgm(h1_base+'input_space.dat')
    homology2_input = per.read_pdgm(h2_base+'input_space.dat')

    homology0_regular = per.read_pdgm(h0_base+'regular.dat')
    homology1_regular = per.read_pdgm(h1_base+'regular.dat')
    homology2_regular = per.read_pdgm(h2_base+'regular.dat')

    homology0_random = per.read_pdgm(h0_base+'random.dat')
    homology1_random = per.read_pdgm(h1_base+'random.dat')
    homology2_random = per.read_pdgm(h2_base+'random.dat')

    regDH0, regDH1, regDH2 = per.compute_distances(homology0_input,
                                                   homology1_input,
                                                   homology0_regular,
                                                   homology1_regular,
                                                   homology2_input,
                                                   homology2_regular)

    ranDH0, ranDH1, ranDH2 = per.compute_distances(homology0_input,
                                                   homology1_input,
                                                   homology0_random,
                                                   homology1_random,
                                                   homology2_input,
                                                   homology2_random)
    print("=" * 30)
    print("Bootstrap distance (Regular) - H0: %f, H1: %f, H2: %f"
          % (regDH0, regDH1, regDH2))
    print("Bootstrap distance (Random) - H0: %f, H1: %f, H2: %f"
          % (ranDH0, ranDH1, ranDH2))
    print("=" * 30)

    dgm_input = per.read_pdgm(bc_base+'input_space.dat')
    dgm_regular = per.read_pdgm(bc_base+'regular.dat')
    dgm_random = per.read_pdgm(bc_base+'random.dat')

    plot_diagrams(dgm_input, dgm_regular, dgm_random)
    plt.savefig("experiment-MNIST-analysis.pdf")
    plt.show()
