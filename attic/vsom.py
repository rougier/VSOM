# -----------------------------------------------------------------------------
# VSOM (Voronoidal Self Organized Map)
# Copyright (c) 2019 Nicolas P. Rougier
#
# Distributed under the terms of the BSD License.
# -----------------------------------------------------------------------------
import sys
import tqdm
import numpy as np
import scipy.spatial
from math import sqrt, ceil, floor, pi, cos, sin

import scipy.spatial
import networkx as nx
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.collections import LineCollection, PolyCollection



# http://stackoverflow.com/questions/28665491/...
#    ...getting-a-bounded-polygon-coordinates-from-voronoi-cells
def in_box(points, bbox):
    return np.logical_and(
        np.logical_and(bbox[0] <= points[:, 0], points[:, 0] <= bbox[1]),
        np.logical_and(bbox[2] <= points[:, 1], points[:, 1] <= bbox[3]))


def voronoi(points, bbox):
    # See http://stackoverflow.com/questions/28665491/...
    #   ...getting-a-bounded-polygon-coordinates-from-voronoi-cells
    # See also https://gist.github.com/pv/8036995
    
    # Select points inside the bounding box
    i = in_box(points, bbox)

    # Mirror points
    points_center = points[i, :]
    points_left = np.copy(points_center)
    points_left[:, 0] = bbox[0] - (points_left[:, 0] - bbox[0])
    points_right = np.copy(points_center)
    points_right[:, 0] = bbox[1] + (bbox[1] - points_right[:, 0])
    points_down = np.copy(points_center)
    points_down[:, 1] = bbox[2] - (points_down[:, 1] - bbox[2])
    points_up = np.copy(points_center)
    points_up[:, 1] = bbox[3] + (bbox[3] - points_up[:, 1])
    points = np.append(points_center,
                       np.append(np.append(points_left, points_right, axis=0),
                                 np.append(points_down, points_up, axis=0),
                                 axis=0), axis=0)
    # Compute Voronoi
    vor = scipy.spatial.Voronoi(points)
    epsilon = sys.float_info.epsilon

    # Filter regions
    regions = []
    for region in vor.regions:
        flag = True
        for index in region:
            if index == -1:
                flag = False
                break
            else:
                x = vor.vertices[index, 0]
                y = vor.vertices[index, 1]
                if not(bbox[0]-epsilon <= x <= bbox[1]+epsilon and
                       bbox[2]-epsilon <= y <= bbox[3]+epsilon):
                    flag = False
                    break
        if region != [] and flag:
            regions.append(region)
    vor.filtered_points = points_center
    vor.filtered_regions = regions
    return vor


def centroid(V):
    """
    Given an ordered set of vertices V describing a polygon,
    returns the uniform surface centroid.

    See http://paulbourke.net/geometry/polygonmesh/
    """
    A = 0
    Cx = 0
    Cy = 0
    for i in range(len(V)-1):
        s = (V[i, 0]*V[i+1, 1] - V[i+1, 0]*V[i, 1])
        A += s
        Cx += (V[i, 0] + V[i+1, 0]) * s
        Cy += (V[i, 1] + V[i+1, 1]) * s
    Cx /= 3*A
    Cy /= 3*A
    return [Cx, Cy]


def blue_noise(shape, radius, k=30, seed=None):
    """
    Generate blue noise over a two-dimensional rectangle of size (width,height)

    Parameters
    ----------

    shape : tuple
        Two-dimensional domain (width x height) 
    radius : float
        Minimum distance between samples
    k : int, optional
        Limit of samples to choose before rejection (typically k = 30)
    seed : int, optional
        If provided, this will set the random seed before generating noise,
        for valid pseudo-random comparisons.

    References
    ----------

    .. [1] Fast Poisson Disk Sampling in Arbitrary Dimensions, Robert Bridson,
           Siggraph, 2007. :DOI:`10.1145/1278780.1278807`
    """

    def sqdist(a, b):
        """ Squared Euclidean distance """
        dx, dy = a[0] - b[0], a[1] - b[1]
        return dx * dx + dy * dy

    def grid_coords(p):
        """ Return index of cell grid corresponding to p """
        return int(floor(p[0] / cellsize)), int(floor(p[1] / cellsize))

    def fits(p, radius):
        """ Check whether p can be added to the queue """

        radius2 = radius*radius
        gx, gy = grid_coords(p)
        for x in range(max(gx - 2, 0), min(gx + 3, grid_width)):
            for y in range(max(gy - 2, 0), min(gy + 3, grid_height)):
                g = grid[x + y * grid_width]
                if g is None:
                    continue
                if sqdist(p, g) <= radius2:
                    return False
        return True

    # When given a seed, we use a private random generator in order to not
    # disturb the default global random generator
    if seed is not None:
        from numpy.random.mtrand import RandomState
        rng = RandomState(seed=seed)
    else:
        rng = np.random
    
    width, height = shape
    cellsize = radius / sqrt(2)
    grid_width = int(ceil(width / cellsize))
    grid_height = int(ceil(height / cellsize))
    grid = [None] * (grid_width * grid_height)

    p = rng.uniform(0, shape, 2)
    queue = [p]
    grid_x, grid_y = grid_coords(p)
    grid[grid_x + grid_y * grid_width] = p

    while queue:
        qi = rng.randint(len(queue))
        qx, qy = queue[qi]
        queue[qi] = queue[-1]
        queue.pop()
        for _ in range(k):
            theta = rng.uniform(0,2*pi)
            r = radius * np.sqrt(rng.uniform(1, 4))
            p = qx + r * cos(theta), qy + r * sin(theta)
            if not (0 <= p[0] < width and 0 <= p[1] < height) or not fits(p, radius):
                continue
            queue.append(p)
            gx, gy = grid_coords(p)
            grid[gx + gy * grid_width] = p

    return np.array([p for p in grid if p is not None])




class VSOM2:
    """ Self Organizing Map """

    def __init__(self, topology="regular", n=1024, n_neighbour=2):
        """
        Initialize SOM

        type: string
            "regular" or "random"
        n : int
            number of neurons
        ndim: int
            dimension of data to be fed to the SOM
        """

        self.topology = topology
        self.n_neighbour = n_neighbour
        
        if self.topology == "regular":
            n = int(np.ceil(np.sqrt(n)))
            X, Y = np.meshgrid(np.linspace(0, 1, n+2, endpoint=True)[1:-1],
                               np.linspace(0, 1, n+2, endpoint=True)[1:-1])
            P = np.c_[X.ravel(), Y.ravel()]
            D = scipy.spatial.distance.cdist(P,P)
            self.positions = P
            self.distances = D / D.max()
            self.voronoi = voronoi(P, bbox=[0, 1, 0, 1])
            self.edges = np.zeros((n*n*2, 2), dtype=int)
            index = 0
            for i in range(n):
                for j in range(n-1):
                    source, target = i*n+j, i*n+j+1
                    self.edges[index] = source, target
                    index += 1
                    source, target = j*n+i, (j+1)*n+i
                    self.edges[index] = source, target
                    index += 1
            
        else:
            radius = np.sqrt(2/(n*np.pi))
            P = blue_noise((1,1), radius=radius)
            self.voronoi = voronoi(P, bbox=[0, 1, 0, 1])
            
            # for i in range(10):
            #     V = voronoi(P, bbox=[0,1,0,1])
            #     C = []
            #     for region in V.filtered_regions:
            #         vertices = V.vertices[region + [region[0]], :]
            #         C.append(centroid(vertices))
            #     P = np.array(C)
    
            self.positions = P
            self.voronoi = V
            D = scipy.spatial.distance.cdist(P,P)
            sources = np.repeat(np.arange(len(P)),n_neighbour)
            sources = sources.reshape(len(P),n_neighbour)
            targets = np.argsort(D,axis=1)[:,1:n_neighbour+1]
            self.edges = np.c_[sources.ravel(), targets.ravel()]
            C = np.zeros(D.shape, dtype=int)
            C[sources,targets] = 1
            lengths = nx.floyd_warshall_numpy(nx.Graph(C))
            self.distances = np.array(lengths).astype(int)
            self.distances = self.distances/self.distances.max()
            

    def __len__(self):
        """ x.__len__() <==> len(x) """

        return len(self.positions)
    
            
    def learn(self, samples, n=10000,
              sigma=(0.50, 0.01), lrate=(0.50, 0.01), labels=None):
        """ Learn samples """

        t = np.linspace(0, 1, n)

        # We will reshape the final codebook to keep samples shape
        shape = [len(self)] + list(samples.shape[1:])
        
        samples = samples.reshape(len(samples), -1)
        self.codebook = np.zeros((len(self), samples.shape[-1]))
        self.labels = np.zeros(len(self))

        lrate = lrate[0]*(lrate[1]/lrate[0])**t
        sigma = sigma[0]*(sigma[1]/sigma[0])**t
        I = np.random.randint(0, len(samples), n)
        samples = samples[I]
        if labels is not None:
            labels = labels[I]
                        
        for i in tqdm.trange(n):
            # Get random sample
            data = samples[i]

            # Get index of nearest node (minimum distance)
            winner = np.argmin(((self.codebook - data)**2).sum(axis=-1))

            # Gaussian centered on winner
            G = np.exp(-self.distances[winner]**2/sigma[i]**2)

            # Move nodes towards sample according to Gaussian 
            self.codebook -= lrate[i]*G[...,np.newaxis]*(self.codebook - data)

            if labels is not None:
                self.labels -= lrate[i]*G*(self.labels-labels[i])
            # self.labels[winner] = labels[i]

        self.codebook = self.codebook.reshape(shape)


    def test(self, samples, labels=None):
        """ Learn samples """

        samples = samples.reshape(len(samples), -1)
        codebook = self.codebook.reshape((len(self), -1))
        error = 0
        for i in tqdm.trange(len(samples)):
            sample = samples[i]
            winner = np.argmin(((codebook - sample)**2).sum(axis=-1))
            error += ((codebook[i] - sample)**2).sum()
        error /= len(samples)
        return error

            
        # samples = samples.reshape(len(samples), -1)
        # codebook = self.codebook.reshape((len(self), -1))
        # #self.labels = np.zeros(len(self))
        # s = []
        # z = 0
        # for i in tqdm.trange(len(samples)):
        #     sample = samples[i]
        #     label  = labels[i]
        #     winner = np.argmin(((codebook - sample)**2).sum(axis=-1))
        #     s.append(np.abs(label - self.labels[winner]))

        #     if label == int((self.labels[winner])):
        #         z += 1
            
        # print(np.mean(s))
        # print(z/len(samples))
            


    def plot_activation(self, ax, sample, cmap='plasma'):

        codebook = self.codebook.reshape(len(self), -1)
       
        D = -np.sqrt(((codebook - sample.ravel())**2).sum(axis=-1))
        P = self.positions
        
        if self.topology == "random":
            V = self.voronoi
            cmap = matplotlib.cm.get_cmap(cmap)
            norm = matplotlib.colors.Normalize(vmin=D.min(), vmax=D.max())
            segments = []
            for region in V.filtered_regions:
                segments.append(V.vertices[region + [region[0]], :])
            collection = PolyCollection(segments, linewidth=1.0,
                                        edgecolors=cmap(norm(D)),
                                        facecolors=cmap(norm(D)))
            ax.add_collection(collection)

            from scipy.interpolate import griddata
            X, Y = np.linspace(0, 1, 512), np.linspace(0, 1, 512)
            Z = griddata(P, D, (X[None,:], Y[:,None]), method='nearest')
            ax.contour(X, Y, Z, 8, linewidths=0.5, colors='k', alpha=0.75)

        else: # regular
            n = int(np.ceil(np.sqrt(len(self))))
            Z = D.reshape(n,n)
            X, Y = np.linspace(0, 1, n), np.linspace(0, 1, n)
            ax.imshow(Z, cmap=cmap, interpolation='nearest', extent=[0,1,0,1],
                      origin="lower")
            ax.contour(X, Y, Z, 8, linewidths=0.5, colors='k', alpha=0.75)

        if len(sample.shape) == 2:
            rows,cols = sample.shape
            image = np.zeros((rows,cols,4))
            image[:,:,0] = image[:,:,1] = image[:,:,2] = 0
            image[:,:,3] = sample
            image = OffsetImage(image, zoom=1.5, zorder=20,
                                interpolation="nearest")
            box = AnnotationBbox(image, (0.9,0.9), frameon=True)
            ax.add_artist(box)

        ax.set_xlim(0,1), ax.set_ylim(0,1)
        ax.set_xticks([]), ax.set_yticks([])



    def plot_network(self, ax):
        size = 50 * 1000/len(self)
        P,V,E = self.positions, self.voronoi, self.edges
        ax.scatter(P[:,0], P[:,1], s=size,
                   edgecolor="k", facecolor="w", linewidth=1.)
        segments = np.zeros((len(E), 2, 2))
        for i in range(len(E)):
            segments[i] = P[E[i,0]], P[E[i,1]]
        collection = LineCollection(segments, color="k", zorder=-10, lw=1.)
        ax.add_collection(collection)

        segments = []
        for region in V.filtered_regions:
            segments.append(V.vertices[region + [region[0]], :])
        collection = LineCollection(segments, color="k", linewidth=0.5,
                                    zorder=-20, alpha=0.25)
        ax.add_collection(collection)
        ax.set_xlim(0,1), ax.set_ylim(0,1)
        ax.set_xticks([]), ax.set_yticks([])


    def plot_weights(self, ax, cmap='magma', samples=None):
        P,V,E = self.positions, self.voronoi, self.edges

        # Display weights as localized images
        if len(self.codebook.shape) == 3:
            rows,cols = self.codebook.shape[1:]
            segments = []
            for region in V.filtered_regions:
                segments.append(V.vertices[region + [region[0]], :])
                collection = PolyCollection(segments, linewidth=0.25, alpha=1.0,
                                            edgecolors="k", facecolors="w")
            ax.add_collection(collection)

    
            for position, data in zip(P, self.codebook):
                image = np.zeros((rows,cols,4))
                image[:,:,3] = data.reshape(rows,cols)
                image = OffsetImage(image,
                                    zoom=0.5, zorder=20, interpolation="nearest")
                box = AnnotationBbox(image, position, frameon=False)
                ax.add_artist(box)

            ax.set_xlim(0,1), ax.set_ylim(0,1)
            ax.set_xticks([]), ax.set_yticks([])
            return
        
        codebook = self.codebook.reshape(len(self), -1)

        # Display weights as a mesh in data space
        if codebook.shape[-1] == 2:
            size = 50 * 1000/len(self)
            X, Y = codebook[:,0], codebook[:,1]
            ax.scatter(X, Y, s=size, edgecolor="w", facecolor="k", linewidth=1.0)
            ax.scatter(samples[:,0], samples[:,1], s=5,
                       edgecolor="None", facecolor="blue",
                       alpha=0.25, zorder=-30)
    
            segments = np.zeros((len(self.edges), 2, 2))
            for i in range(len(self.edges)): 
                segments[i] = codebook[self.edges[i,0]], codebook[self.edges[i,1]]
            collection = LineCollection(segments, linewidth=0.75,
                                        color='black', zorder=-10, alpha=1.0)
            ax.add_collection(collection)
            ax.set_xlim(-1,1), ax.set_ylim(-1,1)
            ax.set_xticks([]), ax.set_yticks([])
            return

        
        if self.topology == "random":
            # Display weights as voronoi cells + cmap
            if codebook.shape[-1] == 1:
                cmap = matplotlib.cm.get_cmap(cmap)
                norm = matplotlib.colors.Normalize(vmin=0, vmax=1)
                facecolors = edgecolors = cmap(norm(self.codebook.ravel()))
            # Display weights as colored voronoi cells
            elif codebook.shape[-1] == 3:
                facecolors = edgecolors = codebook
            segments = []
            for region in V.filtered_regions:
                segments.append(V.vertices[region + [region[0]], :])
            collection = PolyCollection(segments, linewidth=1.0,
                              edgecolors = edgecolors, facecolors = facecolors)
            ax.add_collection(collection)
        else:
            n = int(np.ceil(np.sqrt(len(self))))
            # Display weights on a regular grid + cmap
            if codebook.shape[-1] == 1:
                ax.imshow(codebook.reshape(n,n), cmap=cmap, origin="lower",
                          interpolation='nearest', extent=[0, 1, 0, 1])
            # Display weights on a colored regular grid
            elif self.codebook.shape[-1] == 3:
                ax.imshow(codebook.reshape(n,n,3), origin="lower",
                          interpolation='nearest', extent=[0, 1, 0, 1])

        ax.set_xlim(0,1), ax.set_ylim(0,1)
        ax.set_xticks([]), ax.set_yticks([])

    
    def plot_letter(self, ax, letter):
        text = ax.text(0.05, 0.05, letter, zorder=20,
                       fontsize=32, fontweight="bold", transform=ax.transAxes)
        text.set_path_effects(
            [path_effects.Stroke(linewidth=2, foreground='white'),
             path_effects.Normal()])


    
class VSOM:
    """ Randomized Self Organizing Map """

    def __init__(self, shape, distance):
        ''' Initialize som '''

        self.codebook = np.random.uniform(0, 1, shape)
        self.labels = np.random.uniform(0, 1, len(self.codebook))
        self.distance = distance / distance.max()

    
    def learn(self, samples, n=10000, sigma=(0.25, 0.01), lrate=(0.5, 0.01)):
        """ Learn samples """

        t = np.linspace(0, 1, n)
        lrate = lrate[0]*(lrate[1]/lrate[0])**t
        sigma = sigma[0]*(sigma[1]/sigma[0])**t
        I = np.random.randint(0, len(samples), n)
        samples = samples[I]

        for i in tqdm.trange(n):
            # Get random sample
            data = samples[i]

            # Get index of nearest node (minimum distance)
            winner = np.argmin(((self.codebook - data)**2).sum(axis=-1))

            # Gaussian centered on winner
            G = np.exp(-self.distance[winner]**2/sigma[i]**2)

            # Move nodes towards sample according to Gaussian 
            self.codebook -= lrate[i]*G[...,np.newaxis]*(self.codebook - data)


