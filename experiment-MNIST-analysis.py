import mnist
import numpy as np
import matplotlib
import matplotlib.pylab as plt

from persistence_diagram import persistence

matplotlib.use('Agg')


if __name__ == '__main__':
    np.random.seed(1)
    num_neurons = 1024

    input_space, labels = mnist.read("training")
    index = np.random.choice(np.arange(0, 50000), num_neurons)
    input_space = input_space[index].reshape(num_neurons, 28*28)

    regular_som = np.load("results/experiment-MNIST-regular.npy")
    random_som = np.load("results/experiment-MNIST-random.npy")
    per = persistence(dimension=728, max_edge_length=728)
    per.compute_persistence(input_space,
                            regular_som,
                            random_som,
                            case='MNIST')
    plt.savefig("experiment-MNIST-analysis.pdf")
