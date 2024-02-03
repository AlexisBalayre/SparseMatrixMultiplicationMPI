#ifndef SPARSEMATRIXDENSEVECTORMULTIPLY_H
#define SPARSEMATRIXDENSEVECTORMULTIPLY_H

#include "MatrixDefinitions.h"

/**
 * @brief Function to execute the sparse matrix-fat vector multiplication using sequential algorithm
 *
 * @param sparseMatrix  Sparse matrix
 * @param fatVector Fat Vector
 * @param vecCols  Number of columns in the Fat Vector
 * @return FatVector  Result of the multiplication
 */
FatVector sparseMatrixFatVectorMultiply(const SparseMatrix &sparseMatrix,
                                            const FatVector &fatVector, int vecCols);

#endif
