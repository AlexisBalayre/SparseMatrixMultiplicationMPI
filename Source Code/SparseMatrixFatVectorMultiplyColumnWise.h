#ifndef SPARSEMATRIXFATVECTORMULTIPLYCOLUMNWIZE_H
#define SPARSEMATRIXFATVECTORMULTIPLYCOLUMNWIZE_H

#include "MatrixDefinitions.h"
#include <iostream> // std::cout

/**
 * @brief Function to execute the sparse matrix-Fat Vector multiplication using column-wise parallel algorithm
 *
 * @param sparseMatrix Sparse matrix
 * @param fatVector Fat Vector
 * @param vecCols Number of columns in the Fat Vector
 * @return FatVector Result of the multiplication
 */
FatVector sparseMatrixFatVectorMultiplyColumnWise(const SparseMatrix &sparseMatrix, const FatVector &fatVector, int vecCols);

#endif
