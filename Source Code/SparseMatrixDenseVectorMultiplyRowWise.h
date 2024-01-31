#ifndef SPARSEMATRIXDENSEVECTORMULTIPLYROWWIZE_H
#define SPARSEMATRIXDENSEVECTORMULTIPLYROWWIZE_H

#include "MatrixDefinitions.h"
#include <iostream> // std::cout

/**
 * @brief Function to multiply a sparse matrix with a dense vector using row-wise distribution
 *
 * @param sparseMatrix  The sparse matrix to be multiplied
 * @param denseVector  The dense vector to be multiplied
 * @param vecCols  Number of columns in the dense vector
 * @return DenseVector Result of the multiplication
 */
DenseVector sparseMatrixDenseVectorMultiplyRowWise(const SparseMatrix &sparseMatrix,
                                                   const DenseVector &denseVector,
                                                   int vecCols);

#endif
