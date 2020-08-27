# -----------------------------------------------------------------------------
# VSOM (Voronoidal Self Organized Map)
# Copyright (c) 2019 Nicolas P. Rougier
#
# Distributed under the terms of the BSD License.
# -----------------------------------------------------------------------------
import os
import struct
import numpy as np
import scipy.spatial
import networkx as nx
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.collections import LineCollection, PolyCollection
from vsom import VSOM, blue_noise, voronoi, centroid


def line(orientation=0, thickness=1, antialias=0):
    x1,y1 = np.cos(orientation), np.sin(orientation)
    x2,y2 = -x1, -y1
    X, Y = np.meshgrid(np.arange(-10,10), np.arange(-10,10))
    d = (np.abs((y2-y1)*X-(x2-x1)*Y+x2*y1-y2*x1) /
         np.sqrt((y2-y1)*(y2-y1) + (x2-x1)*(x2-x1)))
    return ((d-thickness/2) < 0).astype(float)



# -----------------------------------------------------------------------------
if __name__ == '__main__':

    # Parameters
    # ----------
    seed        = 1
    n           = 2000
    radius      = np.sqrt(2/(n*np.pi))
    n_neighbour = 2
    n_epochs    = 25000
    sigma       = 0.50, 0.01
    lrate       = 0.50, 0.01
    rows, cols  = 20,20


    
    # Initialization
    # --------------
    if seed is None:
        seed = np.random.randin(0,1000)
    np.random.seed(seed)
    print("Random seed: {0}".format(seed))
        

    # X, Y = np.meshgrid(np.linspace(0,1,32),
    #                    np.linspace(0,1,32))
    # P = np.c_[X.ravel(),Y.ravel()]
    # distance = scipy.spatial.distance.cdist(P,P)
    # V = voronoi(P, bbox=[0,1,0,1])
        
    
    # Nice uniform random distribution (blue noise)
    # ---------------------------------------------
    P = blue_noise((1,1), radius=radius)
    print("Number of neurons: {0}".format(len(P)))

    
    # Centroidal Voronoi Tesselation (10 iterations)
    # ----------------------------------------------
    for i in range(10):
        V = voronoi(P, bbox=[0,1,0,1])
        C = []
        for region in V.filtered_regions:
            vertices = V.vertices[region + [region[0]], :]
            C.append(centroid(vertices))
        P = np.array(C)


    # Connectivity matrix (C) and distance matrix (D)
    # -----------------------------------------------
    D = scipy.spatial.distance.cdist(P,P)
    sources = np.repeat(np.arange(len(P)),n_neighbour).reshape(len(P),n_neighbour)
    targets = np.argsort(D,axis=1)[:,1:n_neighbour+1]
    edges = np.c_[sources.ravel(), targets.ravel()]
    C = np.zeros(D.shape, dtype=int)
    C[sources,targets] = 1
    lengths = nx.shortest_path_length(nx.Graph(C))
    distance = np.zeros(D.shape, dtype=int)
    for i in range(len(P)):
        for j in range(len(P)):
            distance[i,j] = lengths[i][j]

    
    # Train SOM
    # ---------
    n = 10000
    samples = np.zeros((n,rows*cols))
    for i in range(n):
        orientation = np.random.uniform(0,2*np.pi)
        thickness = np.random.uniform(1,10)
        samples[i] = line(orientation, thickness).ravel()
    
    som = VSOM((len(P),rows*cols), distance)
    som.learn(samples, n_epochs, sigma=sigma, lrate=lrate)


    # Display activation for 6 random points
    # --------------------------------------
    indices = np.random.randint(0,len(samples),6)
    for i in range(6):
        indices[i] = np.random.randint(0,len(samples))
        while samples[indices[i]].sum() > 56:
            indices[i] = np.random.randint(0,len(samples))
    
    fig = plt.figure(figsize=(12,8))
    for i in range(len(indices)):
        ax = plt.subplot(2, 3, i+1, aspect=1)
        data = samples[indices[i]]
        D = -np.sqrt(((som.codebook - data)**2).sum(axis=-1)) 
        cmap = matplotlib.cm.get_cmap('plasma')
        norm = matplotlib.colors.Normalize(vmin=D.min(), vmax=D.max())
        segments = []
        for region in V.filtered_regions:
            segments.append(V.vertices[region + [region[0]], :])
        collection = PolyCollection(segments, linewidth=1.0,
                                    edgecolors=cmap(norm(D)),
                                    facecolors=cmap(norm(D)))
        ax.add_collection(collection)
        text = ax.text(0.05, 0.05, chr(ord("C")+i),
                       fontsize=24, fontweight="bold", transform=ax.transAxes)
        text.set_path_effects([path_effects.Stroke(linewidth=2, foreground='white'),
                               path_effects.Normal()])
        ax.set_xlim(0,1), ax.set_ylim(0,1)
        ax.set_xticks([]), ax.set_yticks([])

        from scipy.interpolate import griddata
        X = np.linspace(0, 1, 512)
        Y = np.linspace(0, 1, 512)
        Z = griddata(P, D, (X[None,:], Y[:,None]), method='nearest')
        ax.contour(X, Y, Z, 8, linewidths=0.5, colors='k', alpha=0.75)

        image = np.zeros((rows,cols,4))
        image[:,:,0] = image[:,:,1] = image[:,:,2] = 0
        image[:,:,3] = data.reshape(rows,cols)
        image = OffsetImage(image, zoom=2, zorder=20, interpolation="nearest")
        # image = OffsetImage(data.reshape(rows,cols), zoom=0.5,
        #                     zorder=-20, cmap='gray_r')
        box = AnnotationBbox(image, (0.9,0.9), frameon=True)
        ax.add_artist(box)

        
    plt.tight_layout()
    #plt.savefig("vsom-image-2.pdf")
    plt.show()

    # Display neural and weight maps
    # ------------------------------
    fig = plt.figure(figsize=(7,7))

    #ax = plt.subplot(1, 3, 1, aspect=.74, axisbelow=False)
    #img = imageio.imread('mucha.png') / 255
    #ax.imshow(img, cmap='gray')
    #ax.set_xticks([]), ax.set_yticks([])
    

    ax = plt.subplot(1, 1, 1, aspect=1, axisbelow=False)
    segments = []
    for region in V.filtered_regions:
        segments.append(V.vertices[region + [region[0]], :])
    collection = PolyCollection(segments, linewidth=0.25, alpha=1.0,
                                edgecolors="0.5", facecolors="w")
    ax.add_collection(collection)
    
    for position, data in zip(P, som.codebook):
        image = np.zeros((rows,cols,4))
        image[:,:,0] = image[:,:,1] = image[:,:,2] = 0
        image[:,:,3] = data.reshape(rows,cols)
        image = OffsetImage(image, zoom=0.45, zorder=20, interpolation="nearest")
        # image = OffsetImage(data.reshape(rows,cols),
        #                     zoom=.75,  zorder=-20, cmap='gray_r')
        # image = OffsetImage(data.reshape(rows,cols,3),
        #                     zoom=.75,  zorder=-20)
        box = AnnotationBbox(image, position, frameon=False)
        ax.add_artist(box)
                                
    ax.set_xlim(0,1), ax.set_ylim(0,1)
    ax.set_xticks([]), ax.set_yticks([])

    # Redraw axis because boxes cover it (and zorder doesn't work)
    ax.plot([0,1,1,0,0],[0,0,1,1,0], c='k', lw=.75, clip_on=False, zorder=20)

    text = ax.text(0.05, 0.05, "B", zorder=20,
                   fontsize=24, fontweight="bold", transform=ax.transAxes)
    text.set_path_effects([path_effects.Stroke(linewidth=2, foreground='white'),
                           path_effects.Normal()])

        
    #ax.imshow(img, cmap='gray')
    #ax.set_xticks([]), ax.set_yticks([])
    # ax.imshow(img, cmap='gray')
    # ax.set_xticks([]), ax.set_yticks([])

    # plt.tight_layout()
    # plt.show()
    plt.tight_layout()
    # plt.savefig("vsom-image-1.pdf")
    plt.show()

    # #ax = plt.subplot(1, 3, 3, aspect=.74, axisbelow=False)
    # img = imageio.imread('mucha.png') / 255
    # img = (img - img.min())/(img.max() - img.min())
    # for i in range(0, img.shape[0] - rows, rows):
    #     for j in range(0, img.shape[1] - cols, cols):
    #         data = img[i:i+rows, j:j+cols].ravel()
    #         winner = np.argmin(((som.codebook - data)**2).sum(axis=-1))
    #         img[i:i+rows, j:j+cols] = som.codebook[winner].reshape(rows,cols)
    # imageio.imwrite("mucha-vsom.png", np.round(img*255).astype(np.uint8))
    
