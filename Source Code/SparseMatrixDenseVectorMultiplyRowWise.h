#ifndef SPARSEMATRIXDENSEVECTORMULTIPLYROWWIZE_H
#define SPARSEMATRIXDENSEVECTORMULTIPLYROWWIZE_H

#include "MatrixDefinitions.h"

/**
 * @brief Function to multiply a sparse matrix with a dense vector using row-wise distribution
 *
 * @param sparseMatrix  The sparse matrix to be multiplied
 * @param denseVector  The dense vector to be multiplied
 * @param numRows   Number of rows in the sparse matrix
 * @param numCols  Number of columns in the sparse matrix
 * @param vecCols  Number of columns in the dense vector
 * @return DenseVector Result of the multiplication
 */
DenseVector sparseMatrixDenseVectorMultiplyRowWise(const SparseMatrix &sparseMatrix,
                                                   const DenseVector &denseVector,
                                                   int numRows, int numCols, int vecCols);

#endif
