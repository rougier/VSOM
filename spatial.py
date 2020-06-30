# -----------------------------------------------------------------------------
# Copyright (c) 2019 Nicolas P. Rougier
# Distributed under the terms of the BSD License.
# -----------------------------------------------------------------------------
import sys
import numpy as np
import scipy.spatial
# from scipy.spatial import Voronoi
from shapely.geometry import Polygon
from math import sqrt, ceil, floor, pi, cos, sin

def voronoi_finite_polygons_2d(vor, radius=None):
    """
    Reconstruct infinite voronoi regions in a 2D diagram to finite
    regions.
    Parameters
    ----------
    vor : Voronoi
        Input diagram
    radius : float, optional
        Distance to 'points at infinity'.
    Returns
    -------
    regions : list of tuples
        Indices of vertices in each revised Voronoi regions.
    vertices : list of tuples
        Coordinates for revised Voronoi vertices. Same as coordinates
        of input vertices, with 'points at infinity' appended to the
        end.
    """

    if vor.points.shape[1] != 2:
        raise ValueError("Requires 2D input")

    new_regions = []
    new_vertices = vor.vertices.tolist()

    center = vor.points.mean(axis=0)
    if radius is None:
        radius = vor.points.ptp().max()*2

    # Construct a map containing all ridges for a given point
    all_ridges = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))

    # Reconstruct infinite regions
    for p1, region in enumerate(vor.point_region):
        vertices = vor.regions[region]

        if all(v >= 0 for v in vertices):
            # finite region
            new_regions.append(vertices)
            continue

        # reconstruct a non-finite region
        ridges = all_ridges[p1]
        new_region = [v for v in vertices if v >= 0]

        for p2, v1, v2 in ridges:
            if v2 < 0:
                v1, v2 = v2, v1
            if v1 >= 0:
                # finite ridge: already in the region
                continue

            # Compute the missing endpoint of an infinite ridge

            t = vor.points[p2] - vor.points[p1] # tangent
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])  # normal

            midpoint = vor.points[[p1, p2]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n
            far_point = vor.vertices[v2] + direction * radius

            new_region.append(len(new_vertices))
            new_vertices.append(far_point.tolist())

        # sort region counterclockwise
        vs = np.asarray([new_vertices[v] for v in new_region])
        c = vs.mean(axis=0)
        angles = np.arctan2(vs[:,1] - c[1], vs[:,0] - c[0])
        new_region = np.array(new_region)[np.argsort(angles)]

        # finish
        new_regions.append(new_region.tolist())

    return new_regions, np.asarray(new_vertices)



def clipped_voronoi(points, bbox=[0,1,0,1]):
    vor = scipy.spatial.Voronoi(points)
    regions, vertices = voronoi_finite_polygons_2d(vor)
    min_x = vor.min_bound[0] - 0.1
    max_x = vor.max_bound[0] + 0.1
    min_y = vor.min_bound[1] - 0.1
    max_y = vor.max_bound[1] + 0.1
    mins = np.tile((min_x, min_y), (vertices.shape[0], 1))
    bounded_vertices = np.max((vertices, mins), axis=0)
    maxs = np.tile((max_x, max_y), (vertices.shape[0], 1))
    bounded_vertices = np.min((bounded_vertices, maxs), axis=0)

    xmin,xmax,ymin,ymax = bbox
    clip = Polygon([(xmin,ymin), (xmin, ymax), (xmax,ymax), (xmax,ymin)])
    polygons = []
    for region in regions:
        polygon = vertices[region]
        poly = Polygon(polygon)
        poly = poly.intersection(clip)
        polygons.append(np.array([p for p in poly.exterior.coords]))
    return polygons


def voronoi(points, bbox):
    # See http://stackoverflow.com/questions/28665491/...
    #   ...getting-a-bounded-polygon-coordinates-from-voronoi-cells
    # See also https://gist.github.com/pv/8036995

    def in_box(points, bbox):
        return np.logical_and(
            np.logical_and(bbox[0] <= points[:, 0], points[:, 0] <= bbox[1]),
            np.logical_and(bbox[2] <= points[:, 1], points[:, 1] <= bbox[3]))
    
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



def blue_noise(shape, radius, k=30):
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

    
    width, height = shape
    cellsize = radius / sqrt(2)
    grid_width = int(ceil(width / cellsize))
    grid_height = int(ceil(height / cellsize))
    grid = [None] * (grid_width * grid_height)

    rng = np.random
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



def distribution(n=1024, dtype="random", jitter=0):
    """
    Generate a 2d uniform distribution
    """

    # Regular jittered grid distribution
    if dtype in "regular":
        n = int(np.ceil(np.sqrt(n)))
        d = 1/(2*n)
        X, Y = np.meshgrid(np.linspace(d, 1-d, n, endpoint=True),
                           np.linspace(d, 1-d, n, endpoint=True))
        P = np.c_[X.ravel(), Y.ravel()]
        P += jitter * np.random.uniform(-1, 1, P.shape)
        P[:,0] = np.minimum(np.maximum(P[:,0], d), 1-d)
        P[:,1] = np.minimum(np.maximum(P[:,1], d), 1-d)
        
        V = clipped_voronoi(P, bbox=[0,1,0,1])
        return P, V 


    # Blue noise distribution
    radius = np.sqrt(2/(n*pi))
    P = blue_noise(shape=(1,1), radius=radius)
    
    # Add or remove points to have exact count
    if len(P) > n:
        P = P[:n]
    elif len(P) < n:
        P = np.concatenate([P, np.random.uniform(0, 1, (n-len(P),2))])
        
    # Lloyd relaxation (because of added or removed points)
    for i in range(10):
        C, V = [], clipped_voronoi(P, bbox=[0,1,0,1])
        for vertices in V:
            C.append(centroid(vertices))
        P = np.array(C)
    return P, V



# -----------------------------------------------------------------------------
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    def plot(ax, P, V):
        ax.scatter(P[:,0], P[:,1], s=25, fc="w", ec="k", lw=1)
        collection = LineCollection(V, color="0.5", linewidth=0.5)
        ax.add_collection(collection)
        ax.set_xlim(0,1), ax.set_xticks([])
        ax.set_ylim(0,1), ax.set_yticks([])
        
    np.random.seed(123)
    plt.figure(figsize=(12,4))
    P, V = distribution(n=256, dtype="regular", jitter=0.00)
    plot(plt.subplot(1, 3, 1, aspect=1), P, V)
    P, V = distribution(n=256, dtype="regular", jitter=0.02)
    plot(plt.subplot(1, 3, 2, aspect=1), P, V)
    P, V = distribution(n=256, dtype="blue")
    plot(plt.subplot(1, 3, 3, aspect=1), P, V)
    plt.tight_layout()
    plt.show()
