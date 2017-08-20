
# Exhaustive batchwise higher-order MMF -- with parallel operations

import numpy as np
import scipy as sp
import matplotlib as mpl

def batchMMF_Exhaus(A, k):
    
    # initializations and checks
    n = np.shape(A)
    if k>5:
        print "The order of the batch MMF is too huge! Reducing it appropriately."
    
    # loading the orthogonal matrices
    #filename = "OrthMats" + str(k) + ".npy"
    #np.load(filename)        
    
    # setting up the number of levels
    L = n-(k-1)
    
    
    
    

A = np.ones((10,10))
batchMMF_Exhaus(A, 4)
