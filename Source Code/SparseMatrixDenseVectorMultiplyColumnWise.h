#ifndef SPARSEMATRIXDENSEVECTORMULTIPLYCOLUMNWIZE_H
#define SPARSEMATRIXDENSEVECTORMULTIPLYCOLUMNWIZE_H

#include "MatrixDefinitions.h"

/**
 * @brief Function to execute the sparse matrix-dense vector multiplication using column-wise parallel algorithm
 *
 * @param sparseMatrix Sparse matrix
 * @param denseVector Dense vector
 * @param numRow Number of rows in the sparse matrix
 * @param numCols Number of columns in the sparse matrix
 * @param vecCols Number of columns in the dense vector
 * @return DenseVector Result of the multiplication
 */
DenseVector sparseMatrixDenseVectorMultiplyColumnWise(const SparseMatrix &sparseMatrix, const DenseVector &denseVector, int numRows, int numCols, int vecCols);

#endif
