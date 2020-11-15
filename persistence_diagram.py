import pickle
import gudhi as gd
# from gudhi.wasserstein import wasserstein_distance


class persistence(object):
    def __init__(self, dimension=2,
                 max_edge_length=1,
                 max_alpha_square=1,
                 is_alpha_simplex_on=True):
        self.dimension = dimension
        self.max_edge_length = max_edge_length
        self.max_alpha_square = max_alpha_square
        self.is_alpha_simplex_on = is_alpha_simplex_on

    def persistence_diagram(self, data):
        cp = data.tolist()
        if self.is_alpha_simplex_on:
            skeleton = gd.AlphaComplex(points=cp, precision='fast')
            alpha_simplex_tree = skeleton.create_simplex_tree(max_alpha_square=self.max_alpha_square)
            bar_codes = alpha_simplex_tree.persistence()
            dim0 = alpha_simplex_tree.persistence_intervals_in_dimension(0)
            dim1 = alpha_simplex_tree.persistence_intervals_in_dimension(1)
            dim2 = alpha_simplex_tree.persistence_intervals_in_dimension(2)
        else:
            skeleton = gd.RipsComplex(points=cp,
                                      max_edge_length=self.max_edge_length)
            rips_simplex_tree = skeleton.create_simplex_tree(max_dimension=self.dimension)
            bar_codes = rips_simplex_tree.persistence()
            dim0 = rips_simplex_tree.persistence_intervals_in_dimension(0)
            dim1 = rips_simplex_tree.persistence_intervals_in_dimension(1)
            dim2 = rips_simplex_tree.persistence_intervals_in_dimension(2)
        return bar_codes, dim0, dim1, dim2

    def compute_persistence(self,
                            point_cloud,
                            case='Persistence Homology'):
        bc, h0, h1, h2 = self.persistence_diagram(point_cloud)
        # bc_regular, h0_reg, h1_reg, _ = self.persistence_diagram(regular)
        # bc_random, h0_ran, h1_ran, _ = self.persistence_diagram(random)

        self.store_pdgm(bc,
                        "./results/barcode-experiment-"+case+".dat")
        self.store_pdgm(h0,
                        "./results/homology0-experiment-"+case+".dat")
        self.store_pdgm(h1,
                        "./results/homology1-experiment-"+case+".dat")
        self.store_pdgm(h2,
                        "./results/homology2-experiment-"+case+".dat")

    def compute_distances(self,
                          homology0_X,
                          homology1_X,
                          homology0_Y,
                          homology1_Y,
                          homology2_X=None,
                          homology2_Y=None):
        inf = float('inf')
        homology0_X = [h for h in homology0_X if h[1] != inf]
        homology1_X = [h for h in homology1_X if h[1] != inf]
        homology0_Y = [h for h in homology0_Y if h[1] != inf]
        homology1_Y = [h for h in homology1_Y if h[1] != inf]

        DH0 = gd.bottleneck_distance(homology0_X, homology0_Y, e=0)
        DH1 = gd.bottleneck_distance(homology1_X, homology1_Y, e=0)
        DH2 = gd.bottleneck_distance(homology2_X, homology2_Y, e=0)
        return DH0, DH1, DH2

    def store_pdgm(self, dgm, fname='example.data'):
        with open(fname, "wb") as f:
            pickle.dump(dgm, f)

    def read_pdgm(self, fname):
        with open(fname, "rb") as f:
            dgm = pickle.load(f)
        return dgm
