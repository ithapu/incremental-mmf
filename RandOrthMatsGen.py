# Generates a random n x n orthogonal real matrix

import numpy as np
import os as os

def rand_orth_mat(n):

    # initialize
    M = np.zeros((n, n))
    tol = 10 ** -6

    # Gram-Schmidt on random column vectors
    M[:, 0] = np.random.normal(0, 1, n)
    M[:, 0] = M[:, 0] / np.linalg.norm(M[:, 0])

    for col in range(1, n):
        nrm = 0
        while nrm < tol:
            vi = np.transpose([np.random.normal(0, 1, n)])
            vi = vi - np.dot(M[:, 0:col + 1], np.dot(np.transpose(M[:, 0:col + 1]), vi))
            nrm = np.linalg.norm(vi)
        M[:, col] = np.reshape(vi, len(vi)) / np.linalg.norm(vi)
    M = np.matrix(M)

    return M

# the order of the orthogonal matrices
K = range(2, 10)  # kernel size ranging from 2 to 8

for k in K:
    # the number of orthogonal matrices
    searchnum = 2 * (10 ** min([k, 4]))

    # initialize the big ndarray
    Os = np.zeros((k, k, searchnum))
    Os_tran = np.zeros((k, k, searchnum))
    for s in range(1, searchnum + 1):
        O = rand_orth_mat(k)
        Os[:, :, s - 1] = O
        Os_tran[:, :, s - 1] = O.T

    # save the orthogonal matrices
    filename = os.getcwd() + "/OrthMats/" + "OrthMats" + str(k) + ".npz"
    np.savez(filename, Os=Os, Os_tran=Os_tran, searchnum=searchnum)

