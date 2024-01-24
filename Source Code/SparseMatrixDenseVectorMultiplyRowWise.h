#ifndef SPARSEMATRIXDENSEVECTORMULTIPLYROWWIZE_H
#define SPARSEMATRIXDENSEVECTORMULTIPLYROWWIZE_H

#include "MatrixDefinitions.h"

// Function to execute the sparse matrix-dense vector multiplication using row-wise parallel algorithm
DenseMatrix sparseMatrixDenseVectorMultiplyRowWise(const SparseMatrix &sparseMatrix,
                                                   const DenseMatrix &denseVector,
                                                   int numRows, int numCols, int vecCols);

#endif
