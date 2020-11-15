import numpy as np
import matplotlib.pylab as plt

from persistence_diagram import persistence
from plot_persistent_homology import plot_diagrams


if __name__ == '__main__':
    np.random.seed(3)
    num_neurons = 4096
    bc_base = './results/barcode-experiment-3D-uniform-'
    h0_base = './results/homology0-experiment-3D-uniform-'
    h1_base = './results/homology1-experiment-3D-uniform-'
    h2_base = './results/homology2-experiment-3D-uniform-'

    input_space = np.random.uniform(0, 1, (num_neurons, 3))
    regular_som = np.load("results/experiment-3D-uniform-regular.npy")
    random_som = np.load("results/experiment-3D-uniform-random.npy")
    data = [input_space, regular_som, random_som]
    case = ['input_space', 'regular', 'random']
    per = persistence(dimension=3, max_edge_length=1, max_alpha_square=2)
    for i, d in enumerate(data):
        per.compute_persistence(d, case='3D-uniform-'+case[i])

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
    plt.savefig("experiment-3D-uniform-analysis.pdf")
    plt.show()
