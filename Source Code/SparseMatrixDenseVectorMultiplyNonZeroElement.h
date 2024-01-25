#ifndef SPARSEMATRIXDENSEVECTORMULTIPLYNONZEROELEMENT_H
#define SPARSEMATRIXDENSEVECTORMULTIPLYNONZEROELEMENT_H

#include "MatrixDefinitions.h"

/**
 * @brief Function to execute the sparse matrix-dense vector multiplication using non-zero element parallel algorithm
 *
 * @param sparseMatrix  Sparse matrix
 * @param denseVector  Dense vector
 * @param numRows  Number of rows in the sparse matrix
 * @param numCols  Number of columns in the sparse matrix
 * @param vecCols  Number of columns in the dense vector
 * @return DenseVector  Result of the multiplication
 */
DenseVector sparseMatrixDenseVectorMultiplyNonZeroElement(const SparseMatrix &sparseMatrix, const DenseVector &denseVector);

#endif
