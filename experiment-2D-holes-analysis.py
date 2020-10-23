import numpy as np
import matplotlib
import matplotlib.pylab as plt

from persistence_diagram import persistence

matplotlib.use('Agg')


if __name__ == '__main__':
    np.random.seed(12345)
    num_neurons = 1024

    input_space = np.random.uniform(0, 1, (num_neurons, 2))
    regular_som = np.load("results/experiment-2D-holes-regular.npy")
    random_som = np.load("results/experiment-2D-holes-random.npy")
    per = persistence(dimension=2, max_edge_length=1)
    per.compute_persistence(input_space,
                            regular_som,
                            random_som,
                            case='2D-holes')
    plt.savefig("experiment-2D-holes-analysis.pdf")
