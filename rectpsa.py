#!/usr/bin/env python
# This Python script implements the algorithm described in [1] for computing
# the ε-pseudospectra.
# Copyright (C) 2018 Georgios Is. Detorakis (gdetor@pm.me)
#
# 1. T.G. Wright and L.N Trefethen, "Pseudospectra of rectangular matrices",
#    IMA Journal of Numerical Analysis, 22, 501--519, 2002.
#
# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 2
# of the License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301,
# USA.
import numpy as np
import scipy as sc
from scipy.sparse.linalg import eigsh


def psa(A, X, Y, num_eigs=2, method='lanczos'):
    """
        This function implements the algorithm for computing the
        ε-pseudospectra of matrix A.

        Args:
            A (ndarray):    The input (m, n) matrix
            X (ndarray):    The abscissa of the grid (complex plane) on which
                            the pseudospectra will be computed
            Y (ndarray):    The ordinate of the grid
            num_eigs (int): Dummy variable (to be removed in the future)
            method (str):   Defines the iteration method (Lanczos or SVD) for
                            computing the singular values

        Return:
            sigma (ndarray): The pseudospectra of matrix A
    """
    m, n = A.shape

    if method == 'lanczos':
        num_eigs = np.ndim(A) - 1
    else:
        num_eigs = A.shape[0]

    if X.size == 0 or Y.size == 0:
        raise "No grid points found!"

    if m >= 2*n:
        I_ = np.zeros((2*n, n))
        np.fill_diagonal(I_[:n, :], 1)
    else:
        I_ = np.zeros((m, n))
        np.fill_diagonal(I_, 1)
    T = np.zeros((m, n))

    if m >= 2*n:
        print("M >= 2N")
        S = np.zeros((2*n, n))
        Q, R = np.linalg.qr(A[n:m, :], mode='complete')
        S[:n, :] = A[:n, :]
        S[n:, :] = R[:n, :]
        T = I_
    else:
        print("M < 2N")
        S = np.zeros((m, n))
        A1, A2 = A[:m-n, :], A[m-n:, :]
        I1, I2 = I_[:m-n, :], I_[m-n:, :]
        S2, T2, _, _, Q, Z = sc.linalg.ordqz(A2, I2, sort='iuc')
        S[:m-n, :] = np.dot(A1, Z)
        S[m-n:, :] = S2
        # S[m-n:, :] = np.dot(np.dot(Q, A2), Z)
        T[:m-n, :] = np.dot(I1, Z)
        T[m-n:, :] = T2
        # T[m-n:, :] = np.dot(np.dot(Q, I2), Z)

    # eps = 10e-1
    zz = X + 1j * Y
    sigma = np.zeros((zz.shape[0], zz.shape[1]))
    sigma_m = np.zeros((zz.shape[0], zz.shape[1]))
    if method == "lanczos":
        for i in range(zz.shape[0]):
            for j in range(zz.shape[1]):
                tmp = zz[i, j] * T - S
                Q, R = np.linalg.qr(tmp, mode='complete')
                R = np.asmatrix(R[:n, :])
                tmp = np.dot(R.H, R)
                s, v = eigsh(tmp, k=num_eigs, which='SA')
                sigma[i, j] = np.sqrt(s.real)
                # sigma[i, j] = 1.0 / np.sqrt(s.min().real+1e-10)
    elif method == 'svd':
        for i in range(zz.shape[0]):
            for j in range(zz.shape[1]):
                tmp = zz[i, j] * T - S
                Q, R = np.linalg.qr(tmp, mode='complete')
                R = np.asmatrix(R[:n, :])
                tmp = np.dot(R.H, R)
                u, s, v = np.linalg.svd(tmp)
                sigma[i, j] = np.sqrt(s.min().real)
                sigma_m[i, j] = np.sqrt(s.max().real)
                # sigma[i, j] = 1.0 / np.sqrt(s.min()+1e-10)
    else:
        raise "No method found!"

    return sigma, sigma_m


def cond(A):
    """
        This function computes the condition number of any matrix A, based on
        the estimation of the l2 norm.

        Args:
            A (ndarray):    A (m, n) matrix

        Return:
            cond (float):   The condintion number of the matrix A
    """
    w, v = np.linalg.eig(np.dot(np.matrix(A).H, A))
    u, s, v = np.linalg.svd(A)
    At = np.linalg.pinv(A)
    res = np.linalg.norm(A, ord=2) * np.linalg.norm(At, ord=2)
    return res
