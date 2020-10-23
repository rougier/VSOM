import numpy as np
import matplotlib
import matplotlib.pylab as plt

from persistence_diagram import persistence

matplotlib.use('Agg')


if __name__ == '__main__':
    np.random.seed(3)
    num_neurons = 256

    input_space = np.random.uniform(0, 1, (num_neurons, 3))
    regular_som = np.load("results/experiment-3D-uniform-regular.npy")[:256]
    random_som = np.load("results/experiment-3D-uniform-random.npy")[:256]
    per = persistence(dimension=3, max_edge_length=1)
    per.compute_persistence(input_space,
                            regular_som,
                            random_som,
                            case='3D-uniform')
    plt.savefig("experiment-3D-uniform-analysis.pdf")
