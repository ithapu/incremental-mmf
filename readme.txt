
Incremental Multi-resolution Matrix Factorization for modeling symmetric martices

------ Main Scripts ------

INCmmf_EXHAUScomp.m : Incremental MMF procedure with initialization based on Exhaustive or Eigen or Random procedure, 
and the incremental addition of one row-at-a-time done using Exhaustive criterion

INCmmf_EIGENcomp.m : Incremental MMF procedure with initialization based on Exhaustive or Eigen or Random procedure, 
and the incremental addition of one row-at-a-time done using Eigen criterion

INCmmf_RANDOMcomp.m : Incremental MMF procedure with initialization based on Exhaustive or Eigen or Random procedure, 
and the incremental addition of one row-at-a-time done using Random criterion

------ Support Scripts ------

batmmf_EXHAUS.m : Batch MMF with Exhaustive search for both the best k-rows/columns and the best rotation

batmmf_EXHAUS_nonpar.m : Same as batmmf_EXHAUS.m, but no parfor like parallel operations ar used

batmmf_EIGEN.m : Batch MMF with Exhaustive search for the best k-rows/columns, 
but an approximate rotation computed using Eigen-computation

batmmf_EIGEN_nonpar.m : Same as batmmf_EIGEN.m, but no parfor like parallel operations ar used

batmmf_RANDOM.m : Batch MMF with Random search (based on normalized inner-product) for the best k-rwos/columns, 
and an approximate rotation computed using Eigen-computation

OrthMatsGen.m : Script for generating the set of Orthogonal rotation matrices (of a given order k) 
which will be used in searching for the best rotation
(IMP: Need to run this before exploring with any other code -- the outputs generated are too large to pre-share as zip/dropbox files)

------ Visualization (Matlab Console) Scripts ------

visualize_mmf.m : Visualizing a given MMF -- the output is the MMF graph

visualize_mmf_INSERT.m : Visualizing a given MMF after a new row/column is inserted -- the output is the MMF graph
