#ifndef SPARSEMATRIXDENSEVECTORMULTIPLYROWWIZE_H
#define SPARSEMATRIXDENSEVECTORMULTIPLYROWWIZE_H

#include "MatrixDefinitions.h"
#include <iostream> // std::cout

/**
 * @brief Function to multiply a sparse matrix with a Fat Vector using row-wise distribution
 *
 * @param sparseMatrix  The sparse matrix to be multiplied
 * @param fatVector  The Fat Vector to be multiplied
 * @param vecCols  Number of columns in the Fat Vector
 * @return FatVector Result of the multiplication
 */
FatVector sparseMatrixFatVectorMultiplyRowWise(const SparseMatrix &sparseMatrix,
                                                   const FatVector &fatVector,
                                                   int vecCols);

#endif
