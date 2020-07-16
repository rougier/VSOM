# -----------------------------------------------------------------------------
# VSOM (Voronoidal Self Organized Map)
# Copyright (c) 2019 Nicolas P. Rougier
#
# Distributed under the terms of the BSD License.
# -----------------------------------------------------------------------------
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.collections import LineCollection, PolyCollection


def network(ax, som):
    """
    Plot network topology in neural space
    """
    P, V, E = som.positions, som.voronoi, som.edges

    size = 50 * 1000/len(P)
    ax.scatter(P[:, 0], P[:, 1], s=size, ec="k", fc="w", lw=.75)
    segments = np.zeros((len(E), 2, 2))
    for i in range(len(E)):
        segments[i] = P[E[i, 0]], P[E[i, 1]]
    collection = LineCollection(segments, color="k", zorder=-10, lw=1)
    ax.add_collection(collection)
    collection = LineCollection(V, color="k", linewidth=1.0,
                                zorder=-20, alpha=0.25)
    ax.add_collection(collection)
    ax.set_xlim(0, 1), ax.set_ylim(0, 1)
    ax.set_xticks([]), ax.set_yticks([])


def letter(ax, letter):
    """
    Plot a single large letter on the bottom left of the axes
    """
    text = ax.text(0.05, 0.05, letter, zorder=1000,
                   fontsize=32, fontweight="bold", transform=ax.transAxes)
    text.set_path_effects(
        [path_effects.Stroke(linewidth=2, foreground='white'),
         path_effects.Normal()])


def activation(ax, som, sample, cmap='plasma'):
    """
    Plot network activation relative to given sample
    """

    P, V, E = som.positions, som.voronoi, som.edges
    codebook = som.codebook["X"]
    D = -np.sqrt(((codebook - sample)**2).sum(axis=-1))

    cmap = matplotlib.cm.get_cmap(cmap)
    norm = matplotlib.colors.Normalize(vmin=D.min(), vmax=D.max())
    collection = PolyCollection(V, linewidth=1.0,
                                edgecolors=cmap(norm(D)),
                                facecolors=cmap(norm(D)))
    ax.add_collection(collection)
    from scipy.interpolate import griddata
    X, Y = np.linspace(0, 1, 512), np.linspace(0, 1, 512)
    Z = griddata(P, D, (X[None, :], Y[:, None]), method='nearest')
    ax.contour(X, Y, Z, 8, linewidths=0.5, colors='k', alpha=0.75)

    # if len(sample.shape) == 2:
    #     rows,cols = sample.shape
    #     image = np.zeros((rows,cols,4))
    #     image[:,:,0] = image[:,:,1] = image[:,:,2] = 0
    #     image[:,:,3] = sample
    #     image = OffsetImage(image, zoom=1.5, zorder=20,
    #                         interpolation="nearest")
    #     box = AnnotationBbox(image, (0.9,0.9), frameon=True)
    #     ax.add_artist(box)

    ax.set_xlim(0, 1), ax.set_ylim(0, 1)
    ax.set_xticks([]), ax.set_yticks([])


def weights_1D(ax, som, cmap='gray'):
    P, V, E = som.positions, som.voronoi, som.edges
    codebook = som.codebook["X"].ravel()
    cmap = matplotlib.cm.get_cmap(cmap)
    norm = matplotlib.colors.Normalize(vmin=0, vmax=1)
    facecolors = edgecolors = cmap(norm(codebook))
    collection = PolyCollection(
        V, linewidth=1.0, edgecolors=edgecolors, facecolors=facecolors)
    ax.add_collection(collection)
    ax.set_xlim(0, 1), ax.set_ylim(0, 1)
    ax.set_xticks([]), ax.set_yticks([])


def weights_2D(ax, som, X):
    P = som.codebook["X"].reshape(len(som.codebook), 2)
    E = som.edges
    size = 50 * 1000/len(P)
    ax.scatter(P[:, 0], P[:, 1], s=size, ec="k", fc="w", lw=.75, zorder=50)
    ax.scatter(X[:, 0], X[:, 1], s=5, ec="b", fc="b", alpha=.1, zorder=-50)
    segments = np.zeros((len(E), 2, 2))
    for i in range(len(E)):
        segments[i] = P[E[i, 0]], P[E[i, 1]]
    collection = LineCollection(segments, color="k", zorder=-10, lw=1)
    ax.add_collection(collection)
    ax.set_xlim(0, 1), ax.set_ylim(0, 1)
    ax.set_xticks([]), ax.set_yticks([])


def weights_3D(ax, som):
    P, V, E = som.positions, som.voronoi, som.edges
    codebook = som.codebook["X"]
    facecolors = edgecolors = codebook
    collection = PolyCollection(
        V, linewidth=1.0, edgecolors=edgecolors, facecolors=facecolors)
    ax.add_collection(collection)
    ax.set_xlim(0, 1), ax.set_ylim(0, 1)
    ax.set_xticks([]), ax.set_yticks([])


def weights_img(ax, som, shape, inverse=False, zoom=0.5):
    P, V, E = som.positions, som.voronoi, som.edges
    codebook = som.codebook["X"]
    if inverse:
        codebook = 1-codebook

    rows, cols = shape
    collection = PolyCollection(V, linewidth=0.25, alpha=1.0,
                                edgecolors="k", facecolors="w")
    ax.add_collection(collection)
    for position, data in zip(P, codebook):
        image = np.zeros((rows, cols, 4))
        image[:, :, 3] = data.reshape(rows, cols)
        image = OffsetImage(image,
                            zoom=zoom, zorder=20, interpolation="nearest")
        box = AnnotationBbox(image, position, frameon=False)
        ax.add_artist(box)

    ax.set_xlim(0, 1), ax.set_ylim(0, 1)
    ax.set_xticks([]), ax.set_yticks([])
