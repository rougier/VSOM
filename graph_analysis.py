import numpy as np
import networkx as nx
import matplotlib.pylab as plt

# G = nx.Graph(C)


def draw_graph(G):
    """ Draw graph G """
    nx.draw(G)


def graph_laplacian(G, weights=None):
    """ Compute the Laplacian of graph G and its spectrum """
    laplacian = nx.laplacian_matrix(G, )
    spectrum = nx.laplacian_spectrum(G)
    w, v = np.linalg.eig(laplacian.todense())
    return laplacian.todense(), spectrum


def graph_clustering_coef(G):
    """ Computes the clustering coefficient for graph G. Essentialy is the
    probability of two neighbors of a randomly selected node link to each
    other.
    """
    coef = nx.clustering(G)
    avg_coef = nx.average_clustering(G)
    return avg_coef, coef


def graph_agv_path_length(G):
    """ Computes the average shortest path length for graph G. This is the
     average number of steps along the shortest paths for all possible pairs of
     network nodes.
    """
    return nx.average_shortest_path_length(G)


def degree_distribution(G, bins=30, is_plot=True):
    degrees = [G.degree(n) for n in G.nodes()]
    hist, bin_edges = np.histogram(degrees)

    fig = plt.figure(figsize=(11, 11))
    ax = fig.add_subplot(111)
    ax.hist(degrees, bins=bins)
    return hist, bin_edges
