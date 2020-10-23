import pickle
import gudhi as gd
# from gudhi.wasserstein import wasserstein_distance


class persistence(object):
    def __init__(self, dimension=2,
                 max_edge_length=1,
                 is_alpha_simplex_on=False):
        self.dimension = dimension
        self.max_edge_length = max_edge_length
        self.is_alpha_simplex_on = is_alpha_simplex_on

    def persistence_diagram(self, data):
        cp = data.tolist()
        if self.is_alpha_simplex_on:
            skeleton = gd.AlphaComplex(points=cp)
            alpha_simplex_tree = skeleton.create_simplex_tree()
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
                            input_space,
                            regular,
                            random,
                            case='Persistence Homology'):
        bc_input, h0_inp, h1_inp, _ = self.persistence_diagram(input_space)
        bc_regular, h0_reg, h1_reg, _ = self.persistence_diagram(regular)
        bc_random, h0_ran, h1_ran, _ = self.persistence_diagram(random)

        self.store_pdgm(bc_input,
                        "./results/input-barcode-experiment-"+case+".dat")
        self.store_pdgm(bc_regular,
                        "./results/regular-barcode-experiment-"+case+".dat")
        self.store_pdgm(bc_random,
                        "./results/random-barcode-experiment-"+case+".dat")

        print(30*"*")
        print("Regular Bottleneck DH0: ", gd.bottleneck_distance(h0_inp,
                                                                 h0_reg))
        print("Random Bottleneck DH0: ", gd.bottleneck_distance(h0_inp,
                                                                h0_ran))

        print("Regular Bottleneck DH1: ", gd.bottleneck_distance(h1_inp,
                                                                 h1_reg))
        print("Random Bottleneck DH1: ", gd.bottleneck_distance(h1_inp,
                                                                h1_ran))

        # print("Regular Wasserstein DH0: ", wasserstein_distance(iX0, iY0))
        # print("Random Wasserstein DH0: ", wasserstein_distance(iX0, iZ0))

        # print("Regular Wasserstein DH1: ", wasserstein_distance(iX1, iY1))
        # print("Random Wasserstein DH1: ", wasserstein_distance(iX1, iZ1))
        print(30*"*")

    def store_pdgm(self, dgm, fname='example.data'):
        with open(fname, "wb") as f:
            pickle.dump(dgm, f)
