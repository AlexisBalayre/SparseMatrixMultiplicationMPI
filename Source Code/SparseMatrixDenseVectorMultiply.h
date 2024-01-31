#ifndef SPARSEMATRIXDENSEVECTORMULTIPLY_H
#define SPARSEMATRIXDENSEVECTORMULTIPLY_H

#include "MatrixDefinitions.h"

/**
 * @brief Function to execute the sparse matrix-dense vector multiplication using sequential algorithm
 *
 * @param sparseMatrix  Sparse matrix
 * @param denseVector Dense vector
 * @param vecCols  Number of columns in the dense vector
 * @return DenseVector  Result of the multiplication
 */
DenseVector sparseMatrixDenseVectorMultiply(const SparseMatrix &sparseMatrix,
                                            const DenseVector &denseVector, int vecCols);

#endif
