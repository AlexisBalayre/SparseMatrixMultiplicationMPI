#ifndef SPARSEMATRIXDENSEVECTORMULTIPLYNONZEROELEMENT_H
#define SPARSEMATRIXDENSEVECTORMULTIPLYNONZEROELEMENT_H

#include "MatrixDefinitions.h"
#include <iostream> // std::cout

/**
 * @brief Function to execute the sparse matrix-dense vector multiplication using non-zero element parallel algorithm
 *
 * @param sparseMatrix  Sparse matrix
 * @param denseVector  Dense vector
 * @param vecCols  Number of columns in the dense vector
 * @return DenseVector  Result of the multiplication
 */
DenseVector sparseMatrixDenseVectorMultiplyNonZeroElement(const SparseMatrix &sparseMatrix, const DenseVector &denseVector, int vecCols);

#endif
