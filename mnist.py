# -----------------------------------------------------------------------------
# Copyright (c) 2019 Nicolas P. Rougier
# Distributed under the terms of the BSD License.
# -----------------------------------------------------------------------------
import numpy as np
import os
import sys
import struct

# Read the MNIST dataset (training or testing)


def read(dataset="training", path="MNIST"):
    if dataset == "training":
        filename_img = os.path.join(path, 'train-images-idx3-ubyte')
        filename_lbl = os.path.join(path, 'train-labels-idx1-ubyte')
    elif dataset == "testing":
        filename_img = os.path.join(path, 't10k-images-idx3-ubyte')
        filename_lbl = os.path.join(path, 't10k-labels-idx1-ubyte')
    else:
        raise ValueError("dataset must be 'testing' or 'training'")

    with open(filename_lbl, 'rb') as file:
        magic, count = struct.unpack(">II", file.read(8))
        labels = np.fromfile(file, dtype=np.int8)
        Y = np.zeros((len(labels), 10))
        for i in range(len(labels)):
            Y[i, labels[i]] = 1
        labels = Y

    with open(filename_img, 'rb') as file:
        magic, count, rows, cols = struct.unpack(">IIII", file.read(16))
        images = np.fromfile(file, dtype=np.uint8)
        images = images.reshape(count, rows, cols)
        images = (images-images.min())/(images.max()-images.min())

    return images, labels
