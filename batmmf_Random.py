# Random batchwise higher-order MMF -- without parallel operations

import numpy as np
import os as os
import matplotlib.pyplot as plt

def thefunction(A, k, file):

    # initializations and checks
    n = A.shape[0]
    if k > 9:
        print "The order of the batch MMF is too huge! Reducing it appropriately."

    # loading the orthogonal matrices
    filename = os.getcwd() + "/OrthMats/" + "OrthMats" + str(k) + ".npz"
    orthmats = np.load(filename)

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

        # this is an random MMF -- i.e., the best k-tuple to rotate is simply chosen in a random fashion
        # hence there is no combinatorial search -- we start by generating a random 'first' scaling index
        rp = np.random.permutation(len(S_lprev))
        combs = np.zeros(k, dtype=np.int)
        combs[0] = S_lprev[rp[0]]

        # using this random scaling index we then select the remaining based on best normalized inner product
        # not entirely random hence -- this normalized inner product chooses the most-correlated (approximately)
        # entities in some sense
        checkrest = np.setdiff1d(S_lprev, combs[0])
        nips_checkrest = np.zeros(len(checkrest))
        for i in range(0,len(checkrest)):
            nips_checkrest[i] = np.sum( np.multiply( A_lprev[ combs[0], : ], A_lprev[ checkrest[i], : ] ) )
            nips_checkrest[i] = nips_checkrest[i] / np.linalg.norm(A_lprev[ checkrest[i], : ])
        idx = np.argsort(nips_checkrest)
        idx = np.fliplr([idx])[0]
        for i in range(1,k):
            combs[i] = checkrest[ idx[i-1] ]

        # for this randomply chosen k-tuple, we now compute the best rotation
        # this is among the loaded up kth order rotation matrices

        # the block-diagonal submatrix
        A1 = np.matrix( A_lprev[ np.reshape(combs, (k, 1)), combs ] )
        A1mod = np.tile(np.array(A1)[..., None], [1, 1, orthmats['searchnum']])

        # the off-diagonal submatrix
        temp = np.setdiff1d(S_lprev, combs)
        B1 = np.matrix( A_lprev[ np.reshape(combs, (k, 1)), temp ] )
        B = np.dot(B1, B1.T)
        Bmod = np.tile(np.array(B)[..., None], [1, 1, orthmats['searchnum']])

        # the error terms
        term1 = np.einsum( 'ijk,jlk->ilk', np.einsum('ijk,jlk->ilk', orthmats['Os'], A1mod), orthmats['Os_tran'])
        term2 = np.einsum( 'ijk,jlk->ilk', np.einsum('ijk,jlk->ilk', orthmats['Os'], Bmod), orthmats['Os_tran'])

        # some rearranging
        term1pool = np.multiply( np.transpose(term1, (1,0,2)), \
                np.tile( np.array(1-np.identity(k))[..., None], [1, 1, orthmats['searchnum']]) )
        term2pool = np.multiply( term2, np.tile( np.array(np.identity(k))[..., None], [1, 1, orthmats['searchnum']]) )
        term1poolsum = 2 * np.squeeze(np.sum(term1pool, axis=0))
        term2poolsum = 2 * np.squeeze(np.sum(term2pool, axis=0))
        Es_c = term1poolsum + term2poolsum
        Es_c = np.matrix(Es_c.T)

        # computing the best rotation and storing its details
        pos1, pos2 = np.where(Es_c == np.amin(Es_c))
        min_Os_Cs_ind = pos1[0]
        wavepos_ind = pos2[0]

        # now that the best rotation has been computed -- set that appropriately in the factorization bases
        # and compute things -- the resulting compression, the resulting wavelet and scaling indices
        # set the stage for next level
        U_l = np.matrix(np.identity(n))
        I_l = combs
        O_l = np.squeeze(orthmats['Os'][:, :, min_Os_Cs_ind])
        for j1 in range(0, k):
            for j2 in range(0, k):
                U_l[I_l[j1], I_l[j2]] = O_l[j1, j2]

        # the compression at this level -- taking A_lprev to A_l using U_l
        A_l = np.dot(U_l, np.dot(A_lprev, U_l.T))
        pickW = wavepos_ind

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
def batmmf_Random(Mat, Ord, file, disp):

    outs = thefunction(Mat, Ord, file)
    filename = os.getcwd() + file + ".npz"
    np.savez(filename, outs=outs)

    # displaying outs if needed
    if disp:
        print ("Scaling Indices %s" % outs["Sids"])
        print ("Wavelet Indices %s" % outs["Wids"])

    return outs

# example run code
# # temp = np.random.rand(25,20)
# # Mat = np.dot(temp, temp.T)
# # Ord = 6
# # batmmf_Random(Mat, Ord, "trying", 1)

###
# END