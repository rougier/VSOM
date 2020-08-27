#!/usr/bin/env python
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Self-organizing map
# Copyright (C) 2011  Nicolas P. Rougier
#
# Distributed under the terms of the BSD License.
# -----------------------------------------------------------------------------
import numpy as np

def fromdistance(fn, shape, center=None, dtype=float):
    def distance(*args):
        d = 0
        for i in range(len(shape)):
            d += ((args[i]-center[i])/float(max(1,shape[i]-1)))**2
        return np.sqrt(d)/np.sqrt(len(shape))
    if center == None:
        center = np.array(list(shape))//2
    return fn(np.fromfunction(distance,shape,dtype=dtype))

def Gaussian(shape,center,sigma=0.5):
    def g(x): return np.exp(-x**2/sigma**2)
    return fromdistance(g,shape,center)

class SOM:
    def __init__(self, *args):
        ''' Initialize som '''
        self.codebook = np.zeros(args)
        self.reset()

    def reset(self):
        ''' Reset weights '''
        self.codebook = np.random.random(self.codebook.shape)

    def activate(self, sample):
        return ((self.codebook-sample)**2).sum(axis=-1)

    def learn(self, samples, epochs=10000,
                    update=None, update_frequency = 1,
                    sigma=(10, 0.001), lrate=(0.5,0.005)):
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
            D = self.activate(data)
            winner = np.unravel_index(np.argmin(D), D.shape)

            # Generate a Gaussian centered on winner
            G = Gaussian(D.shape, winner, sigma)
            G = np.nan_to_num(G)
            G = G.reshape( list(self.codebook.shape[:-1])+[1] )

            # Move nodes towards sample according to Gaussian
            self.codebook -= lrate*G*(self.codebook-data)

            if update and i % update_frequency == 0:
                update()



# -----------------------------------------------------------------------------
if __name__ == '__main__':
    import matplotlib
    matplotlib.use("TkAgg")
    
    import matplotlib.pyplot as plt

    # Neighborhood
    sigma_initial = 0.25
    sigma_final   = 0.025

    # Learning rate
    lrate_initial = 0.5
    lrate_final   = 0.01

    # Data
    samples = np.random.uniform(0,1,(10000,3))

    # SOM
    n = 64
    som = SOM(n,n,3)
    som.codebook[:,:,1] = som.codebook[:,:,0]
    som.codebook[:,:,2] = som.codebook[:,:,0]

    # Learn
    plt.figure(figsize=(8,8))
    plt.ion()
    im = plt.imshow(som.codebook, interpolation='nearest')
    plt.xticks([]), plt.yticks([])

    def update():
        im.set_data(som.codebook)
        plt.draw()

    som.learn(samples, epochs=5000,
              update = update, update_frequency = 1,
              sigma = (sigma_initial, sigma_final),
              lrate = (lrate_initial, lrate_final))

    plt.ioff()
    plt.show()
