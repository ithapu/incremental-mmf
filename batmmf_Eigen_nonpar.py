# Eigen batchwise higher-order MMF -- without parallel operations

import numpy as np
import os as os
import matplotlib.pyplot as plt

def thefunction(A, k, file):

    # initializations and checks
    n = A.shape[0]
    if k > 5:
        print "The order of the batch MMF is too huge! Reducing it appropriately."

    # setting up the number of levels
    L = n - (k - 1)

    # the row/column indices to be resolved
    S = range(0,n)

    # initialize a bunch of things -- the rotation bases, the compressed As, the wavelet and signal indices
    Us = { "Level 1": np.eye(n) } # U^1, U^2 ... U^L
    Acomps = { "Level 1": np.eye(n) } # A_1, A_2 ... A_L
    Sids = { "Level 1": np.zeros((1,k-1)) }
    Wids = {"Level 1": 0 }
    for level in range(2,L+1):
        Us.update({ "Level " + str(level): np.eye(n) })
        Acomps.update({ "Level " + str(level): np.eye(n) })
        Sids.update({ "Level " + str(level): np.zeros((1,k-1)) })
        Wids.update({"Level " + str(level): 0 })

    # initializing the zeroth level compression -- just some notational thing
    A_l = A
    S_l = S

    ## the meta loop over all the levels
    ## main computations of the multi-resolution factorization
    for level in range (1,L+1):

        # print out the level details
        print "---> " + file + "--- Level " + str(level) + "/" + str(L) + "--- Order " + str(k)

        # initiaizing within this level -- staring at the previous level compression and indices set
        A_lprev = A_l # current approximation of the input matrix A
        S_lprev = S_l # indices/dimensions that have been left out -- to be resolved

        # this is an exhaustive MMF -- i.e., the search through all the possible k-tuples is Exhaustive
        # generating the k-tuple indices to compute errors
        sort_S_lprev = np.sort(S_lprev)
        ls = len(sort_S_lprev)
        if k <= ls:
            #
            if k == 2:
                xv, yv = np.meshgrid(range(0,ls), range(0,ls), sparse=False, indexing='ij')
                x = np.reshape(xv, (np.prod(xv.shape), 1))
                y = np.reshape(yv, (np.prod(xv.shape), 1))
                combs = np.concatenate((x, y), axis=1)
            #
            elif k == 3:
                xv, yv, zv = np.meshgrid(range(0,ls), range(0,ls), range(0,ls), sparse=False, indexing='ij')
                x = np.reshape(xv, (np.prod(xv.shape), 1))
                y = np.reshape(yv, (np.prod(xv.shape), 1))
                z = np.reshape(zv, (np.prod(xv.shape), 1))
                combs = np.concatenate( (np.concatenate((x, y), axis=1), z), axis=1)
            #
            elif k == 4:
                xv, yv, zv, uv = np.meshgrid(range(0,ls), range(0,ls), range(0,ls), range(0,ls), \
                                             sparse=False, indexing='ij')
                x = np.reshape(xv, (np.prod(xv.shape), 1))
                y = np.reshape(yv, (np.prod(xv.shape), 1))
                z = np.reshape(zv, (np.prod(xv.shape), 1))
                u = np.reshape(uv, (np.prod(xv.shape), 1))
                combs = np.concatenate( (np.concatenate((x, y), axis=1), np.concatenate((z, u), axis=1)), axis=1)
            #
            elif k == 5:
                xv, yv, zv, uv, vv = np.meshgrid(range(0,ls), range(0,ls), range(0,ls), \
                                                 range(0,ls), range(0,ls), sparse=False, indexing='ij')
                x = np.reshape(xv, (np.prod(xv.shape), 1))
                y = np.reshape(yv, (np.prod(xv.shape), 1))
                z = np.reshape(zv, (np.prod(xv.shape), 1))
                u = np.reshape(uv, (np.prod(xv.shape), 1))
                v = np.reshape(vv, (np.prod(xv.shape), 1))
                combs = np.concatenate( (np.concatenate( (np.concatenate((x, y), axis=1), np.concatenate((z, u), axis=1)), axis=1), \
                                         v), axis=1)
            #
            else:
                raise ValueError('k is too large -- It should have been reduced!')

        if combs.shape[0] != ls ** k:
            raise ValueError('Something wrong with the calculation of exhaustive combinations!')

        # the combinations were created was using a mesh-grid -- hence, there will be a lot of redundancies e.g.,
        # a k-tuple is k indices that are not equal to each other -- this cannot be avoided by mesh-grid
        # we now remove those erroneous cases
        for row in range(combs.shape[0]):
            for col in range(combs.shape[1]):
                combs[row,col] = sort_S_lprev[combs[row,col]]
        #
        if k == 2:
            remind = (combs[:, 0] - combs[:, 1] != 0)
        elif k == 3:
            remind = 1*(combs[:, 0] - combs[:, 1] != 0) + 1*(combs[:, 0] - combs[:, 2] != 0) \
                     + 1*(combs[:, 1] - combs[:, 2] != 0)
        elif k == 4:
            remind = 1*(combs[:, 0] - combs[:, 1] != 0) + 1*(combs[:, 0] - combs[:, 2] != 0) \
                     + 1*(combs[:, 0] - combs[:, 3] != 0) + 1*(combs[:, 1] - combs[:, 2] != 0) \
                     + 1*(combs[:, 1] - combs[:, 3] != 0) + 1*(combs[:, 2] - combs[:, 3] != 0)
        elif k == 5:
            remind = 1*(combs[:, 0] - combs[:, 1] != 0) + 1*(combs[:, 0] - combs[:, 2] != 0) \
                     + 1*(combs[:, 0] - combs[:, 3] != 0) + 1*(combs[:, 0] - combs[:, 4] != 0) \
                     + 1*(combs[:, 1] - combs[:, 2] != 0) + 1*(combs[:, 1] - combs[:, 3] != 0) \
                     + 1*(combs[:, 1] - combs[:, 4] != 0) + 1*(combs[:, 2] - combs[:, 3] != 0) \
                     + 1*(combs[:, 2] - combs[:, 4] != 0) + 1*(combs[:, 3] - combs[:, 4] != 0)
        #
        combs = combs[remind == k*(k-1)/2]
        combs = np.array(np.sort(combs))
        combs_temp = np.ascontiguousarray(combs).view(np.dtype((np.void, combs.dtype.itemsize * combs.shape[1])))
        _, idx = np.unique(combs_temp, return_index=True)
        combs = combs[idx]

        # given the combinations (k-tuples), we now compute the best possible orthogonal matrices for each
        # of the k-tuples via minimizing the factorization errors
        Es_I = 10 ** 10
        bestcomb_ind = 0
        wavepos_ind = []
        bestOrth = np.identity(k)

        # looping through all the k-tuples
        for c in range(combs.shape[0]):

            # the block-diagonal submatrix
            A1 = np.matrix( A_lprev[ np.reshape(combs[c, :], (k, 1)), combs[c, :] ] )

            # the off-diagonal submatrix
            temp = np.setdiff1d(S_lprev, combs[c, :])
            B1 = np.matrix( A_lprev[ np.reshape(combs[c, :], (k, 1)), temp ] )
            B = np.dot(B1, B1.T)

            # computing the rotation matrix -- this simply corresponds to the eigen vectors of A1
            # we are essentially approximating the best rotation via eigen decomposition of the block-diagonal sub matrix
            # instead of explicitly computing/estimating it as in exhaustive way
            _, V = np.linalg.eig( np.dot(A1.T,A1) )

            # the error terms
            term1 = np.dot( np.dot(V, A1), V.T)
            term2 = np.dot( np.dot(V, B), V.T)

            # some rearranging
            term1pool = np.sum( np.multiply( np.dot(term1, term1.T), (1-np.identity(k)) ), axis=0)
            term2pool = np.diag( term2 )
            Es_c = 2*( term1pool + term2pool )

            # checking for the best rotation -- updating the existing best one if the a rotation fomr the new set of errors
            # gives smaller error
            if np.amin(Es_c) < Es_I:
                Es_I = np.amin(Es_c)
                wavepos_ind = np.where(Es_c == np.amin(Es_c))
                bestOrth = V
                bestcomb_ind = c

        # now that the best rotation has been computed -- set that appropriately in the factorization bases
        # and compute things -- the resulting compression, the resulting wavelet and scaling indices
        # set the stage for next level
        U_l = np.matrix(np.identity(n))
        I_l = combs[bestcomb_ind, :]
        O_l = bestOrth
        for j1 in range(0, k):
            for j2 in range(0, k):
                U_l[I_l[j1], I_l[j2]] = O_l[j1, j2]

        # the compression at this level -- taking A_lprev to A_l using U_l
        A_l = np.dot(U_l, np.dot(A_lprev, U_l.T))
        pickW = wavepos_ind[0][0]

        # saving things for the next level i.e., updating the scaling and wavelet indices list and updating the stack
        # of un-attanded rows (removing wavelet index) -- which needs to be worked out in future levels
        Sids["Level " + str(level)] = str(np.setdiff1d(I_l, I_l[pickW]))
        Wids["Level " + str(level)] = str(I_l[pickW])
        S_l = np.setdiff1d(S_lprev, I_l[pickW])
        Us["Level " + str(level)] = U_l
        Acomps["Level " + str(level)] = A_l

    # operations after all the levels are done -- zero out entries in wavelet positions for the final residual
    # that is the signal lost permanently; and create the residual matrix -- done and dusted!
    H = A_l
    dH = np.diag(H)
    H_S_L = H[S_l,S_l]
    H = np.matrix(np.zeros((n,n)))
    H[S_l,S_l] = H_S_L
    H = H + np.diag(dH)

    # done here -- return the outputs
    outs = {"H": H}
    outs["Us"] = Us
    outs["Sids"] = Sids
    outs["Wids"] = Wids
    outs["Acomps"] = Acomps
    outs["L"] = L
    return outs

####
####

# the main outer-function that call the 'thefunction' routine
def batmmf_Eigen_nonpar(Mat, Ord, file, disp):

    outs = thefunction(Mat, Ord, file)
    filename = os.getcwd() + file + ".npz"
    np.savez(filename, outs=outs)

    # displaying outs if needed
    if disp:
        print ("Scaling Indices %s" % outs["Sids"])
        print ("Wavelet Indices %s" % outs["Wids"])
        print ("")

    return outs

# example run code
# # temp = np.random.rand(6,20)
# # Mat = np.dot(temp, temp.T)
# # Ord = 4
# # batmmf_Exhaus_nonpar(Mat, Ord, "trying", 1)

###
# END