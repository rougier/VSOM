# -----------------------------------------------------------------------------
# VSOM (Voronoidal Self Organized Map)
# Copyright (c) 2019 Nicolas P. Rougier
#
# Distributed under the terms of the BSD License.
# -----------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':
    
    I = np.zeros((10,1000))
    I[:] = np.linspace(0,1,1000)
    e = 0.025

    fig = plt.figure(figsize=(10,1))
    
    ax = plt.subplot(1, 1, 1, aspect=e)
    plt.imshow(I, cmap='viridis', extent=[0, 1, 0, e])
    
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, e)
    ax.set_xticks([0,1])
    ax.set_yticks([])

    plt.tight_layout()
    plt.savefig("colormap-viridis.pdf")

    fig = plt.figure(figsize=(10,1))
    ax = plt.subplot(1, 1, 1, aspect=e)
    plt.imshow(I, cmap='plasma', extent=[0, 1, 0, e])
    
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, e)
    ax.set_xticks([0,1])
    ax.set_yticks([])

    plt.tight_layout()
    plt.savefig("colormap-plasma.pdf")

    
