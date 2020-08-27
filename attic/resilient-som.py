#!/bin/env python
import sys
import numpy as np
import scipy.spatial
import networkx as nx
from math import sqrt, ceil, floor, pi, cos, sin


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



class VOROSOM:
    ''' Voronoidal self organizing map '''

    def __init__(self, shape, topology):
        ''' Initialize som '''
        self.codebook = np.zeros(shape)

        n = len(self.codebook)
        S = nx.shortest_path_length(nx.Graph(topology))
        self.distance = np.zeros((n,n), dtype=int)
        for i in range(n):
            for j in range(n):
                try:
                    self.distance[i,j] = S[i][j]
                except:
                    print("Cannot connect {0} to {1}".format(i,j))
        self.reset()

        
    def reset(self):
        ''' Reset weights '''
        self.codebook = np.random.random(self.codebook.shape)

        
    def learn(self, samples, epochs=10000, sigma=(10, 0.001), lrate=(0.5,0.005)):
        ''' Learn samples '''
        sigma_i, sigma_f = sigma
        lrate_i, lrate_f = lrate

        for i in range(epochs):
            # Adjust learning rate and neighborhood
            t = i/float(epochs)
            lrate = lrate_i*(lrate_f/float(lrate_i))**t
            sigma = sigma_i*(sigma_f/float(sigma_i))**t

            # Get random sample
            index = np.random.randint(0,samples.shape[0])
            data = samples[index]

            # Get index of nearest node (minimum distance)
            D = ((self.codebook-data)**2).sum(axis=-1)
            winner = np.argmin(D)

            # Generate a Gaussian centered on winner
            G = np.exp(-self.distance[winner]**2/sigma**2)

            # Move nodes towards sample according to Gaussian 
            delta = self.codebook - data
            for i in range(self.codebook.shape[-1]):
                self.codebook[...,i] -= lrate * G * delta[...,i]





# -----------------------------------------------------------------------------
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    # Nice uniform random distribution (blue noise)
    P = blue_noise((1,1), 0.05)
    print("Number of neurons: {0}".format(len(P)))

    # Centroidal Voronoi Tesselation (10 iterations)
    for i in range(10):
        V = voronoi(P, bbox=[0.0, 1.0, 0.0, 1.0])
        C = []
        for region in V.filtered_regions:
            vertices = V.vertices[region + [region[0]], :]
            C.append(centroid(vertices))
        P = np.array(C)

    # Connecticity matrix (C) and distance matrix (D)
    p = 4
    D = scipy.spatial.distance.cdist(P,P)
    S = np.repeat(np.arange(len(P)),p).reshape(len(P),p)
    T = np.argsort(D,axis=1)[:,:p]
    L = np.c_[S.ravel(), T.ravel()]
    C = np.zeros(D.shape, dtype=int)
    C[S,T] = 1


    som = VOROSOM((len(P),2), C)

    n = 10000
    # samples = np.random.random((n,2))
    # samples = np.random.normal(.5,.175,(n,2))
    T = np.random.uniform(0, 2*np.pi, n)
    R = np.sqrt(np.random.uniform(0.50**2, 1.0**2, n))
    samples = np.c_[R*np.cos(T), R*np.sin(T)]

    som.learn(samples)

    fig = plt.figure(figsize=(14,7))
    ax = plt.subplot(1, 2, 2, aspect=1)
    X, Y = som.codebook[:,0], som.codebook[:,1]
    ax.scatter(X, Y, s=30, edgecolor="w", facecolor="k", linewidth=1.0)
    ax.scatter(samples[:,0], samples[:,1], s=5,
               edgecolor="None", facecolor="blue", alpha=0.25, zorder=-30)
    
    segments = np.zeros((len(L), 2, 2))
    for i in range(len(L)): 
        segments[i] = som.codebook[L[i,0]], som.codebook[L[i,1]]
    collection = LineCollection(segments, linewidth=0.75,
                                color='black', zorder=-10, alpha=1.0)
    ax.add_collection(collection)
    
#    segments = []
#    V = voronoi(som.codebook, bbox=[0,1,0,1])
#    for region in V.filtered_regions:
#        segments.append(V.vertices[region + [region[0]], :])
#    collection = LineCollection(segments, color='black', linewidth=0.5,
#                                zorder=-20, alpha=0.25)
#    ax.add_collection(collection)
#    ax.set_xlim(-0.025,1.025), ax.set_ylim(-0.025,1.025)
    ax.set_xticks([]), ax.set_yticks([])

    #plt.show()
    
    
    # SPL = nx.shortest_path_length(nx.Graph(C))
    # DT = np.zeros(D.shape, dtype=int)
    # for i in range(len(P)):
    #     for j in range(len(P)):
    #         try:
    #             DT[i,j] = SPL[i][j]
    #         except:
    #             print("Cannot connect {0} to {1}".format(i,j))

    # fig = plt.figure(figsize=(8,6.5))
    # plt.imshow(DT)
    # plt.colorbar()
    # plt.show()
    

    # fig = plt.figure(figsize=(8,8))
    ax = plt.subplot(1, 2, 1, aspect=1)
    X, Y = P[:,0], P[:,1]
    ax.scatter(X, Y, s=25, edgecolor="k", facecolor="w", linewidth=1.0)
    
    segments = np.zeros((len(L), 2, 2))
    for i in range(len(L)): #line in L:
        segments[i] = P[L[i,0]], P[L[i,1]]
    collection = LineCollection(segments, color='black', zorder=-10, alpha=0.5)
    ax.add_collection(collection)

    segments = []
    V = voronoi(P, bbox=[0,1,0,1])
    for region in V.filtered_regions:
        segments.append(V.vertices[region + [region[0]], :])
    collection = LineCollection(segments, color='black', linewidth=0.5,
                                zorder=-20, alpha=0.25)
    ax.add_collection(collection)

    ax.set_xlim(-0.025,1.025), ax.set_ylim(-0.025,1.025)
    ax.set_xticks([]), ax.set_yticks([])

    plt.tight_layout()
    plt.show()
    
