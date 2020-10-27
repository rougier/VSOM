import numpy as np
# import matplotlib
import matplotlib.pylab as plt

from persistence_diagram import persistence
from plot_persistent_homology import plot_diagrams

# matplotlib.use('Agg')


if __name__ == '__main__':
    np.random.seed(3)
    num_neurons = 1024
    bc_base = './results/barcode-experiment-2D-torus-'
    h0_base = './results/homology0-experiment-2D-torus-'
    h1_base = './results/homology1-experiment-2D-torus-'

    T = np.random.uniform(0.0, 2.0*np.pi, num_neurons)
    R = np.sqrt(np.random.uniform(0.50**2, 1.0**2, num_neurons))
    X = 0.5 + np.c_[R*np.cos(T), R*np.sin(T)]/2
    input_space = X

    regular_som = np.load("results/experiment-2D-torus-regular.npy")
    random_som = np.load("results/experiment-2D-torus-random.npy")
    data = [input_space, regular_som, random_som]
    case = ['input_space', 'regular', 'random']
    per = persistence(dimension=2, max_edge_length=1, max_alpha_square=2)
    for i, d in enumerate(data):
        per.compute_persistence(d, case='2D-torus-'+case[i])

    homology0_input = per.read_pdgm(h0_base+'input_space.dat')
    homology1_input = per.read_pdgm(h1_base+'input_space.dat')

    homology0_regular = per.read_pdgm(h0_base+'regular.dat')
    homology1_regular = per.read_pdgm(h1_base+'regular.dat')

    homology0_random = per.read_pdgm(h0_base+'random.dat')
    homology1_random = per.read_pdgm(h1_base+'random.dat')

    regDH0, regDH1 = per.compute_distances(homology0_input,
                                           homology1_input,
                                           homology0_regular,
                                           homology1_regular)

    ranDH0, ranDH1 = per.compute_distances(homology0_input,
                                           homology1_input,
                                           homology0_random,
                                           homology1_random)
    print("=" * 30)
    print("Bootstrap distance (Regular) - H0: %f, H1 %f" % (regDH0, regDH1))
    print("Bootstrap distance (Random) - H0: %f, H1 %f" % (ranDH0, ranDH1))
    print("=" * 30)

    dgm_input = per.read_pdgm(bc_base+'input_space.dat')
    dgm_regular = per.read_pdgm(bc_base+'regular.dat')
    dgm_random = per.read_pdgm(bc_base+'random.dat')

    plot_diagrams(dgm_input, dgm_regular, dgm_random)
    plt.savefig("experiment-2D-torus-analysis.pdf")
    plt.show()
