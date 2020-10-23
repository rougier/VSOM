import numpy as np
import matplotlib
import matplotlib.pylab as plt

from persistence_diagram import persistence

matplotlib.use('Agg')


if __name__ == '__main__':
    np.random.seed(1)
    num_neurons, n_samples = 1024, 1024

    T = np.random.uniform(0.0, 2.0*np.pi, n_samples)
    R = np.sqrt(np.random.uniform(0.50**2, 1.0**2, n_samples))

    input_space = 0.5 + np.c_[R*np.cos(T), R*np.sin(T)]/2
    regular_som = np.load("results/experiment-2D-torus-regular.npy")
    random_som = np.load("results/experiment-2D-torus-random.npy")
    per = persistence(dimension=2, max_edge_length=1)
    per.compute_persistence(input_space,
                            regular_som,
                            random_som,
                            case='2D-torus')
    plt.savefig("experiment-2D-torus-analysis.pdf")
